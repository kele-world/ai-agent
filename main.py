"""
Agent 服务入口（FastAPI）。

对外提供 HTTP 接口，业务系统（比如 Java 后端）通过 HTTP 调用。
Python 只管 AI 编排，业务逻辑留在各自的服务里。

接口：
    GET  /health              健康检查（含配置摘要）
    GET  /agent/tools         查看可用工具及权限级别
    POST /agent/run           执行 Agent
    GET  /metrics             运行统计（累计请求数、token 消耗）
"""

import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agent import get_agent
from config import agent_config, print_summary, server_config
from memory import memory
from tools import ALL_TOOLS, get_tool_level, get_tools_description

# Agent 实例
_agent = None
_init_error: Optional[str] = None

# ---- 简单的内存态限流 ----
# 生产环境应该用 Redis，这里单机演示够用
_rate_window: Dict[str, deque] = defaultdict(deque)

# ---- 运行统计 ----
_metrics = {
    "total_requests": 0,
    "total_tokens": 0,
    "total_elapsed": 0.0,
    "stopped_by": defaultdict(int),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化 Agent"""
    global _agent, _init_error
    print_summary()
    try:
        _agent = get_agent()
        print("\n[OK] Agent 初始化完成\n")
        print("[可用工具]")
        print(get_tools_description())
    except Exception as exc:
        _init_error = str(exc)
        print(f"\n[ERROR] Agent 初始化失败: {exc}")
    yield


app = FastAPI(
    title="校园购物助手 Agent",
    version="1.0.0",
    description="基于 Function Calling + ReAct 的多步推理 Agent",
    lifespan=lifespan,
)

# 前端是本地打开的 HTML（file:// 协议），和后端不同源，必须允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 限流中间件 ====================

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """
    简单限流：单个 IP 每分钟最多 N 次请求。

    Agent 每次调用都烧 token，被刷会很快耗尽额度。
    这里是进程内计数，多实例部署要换成 Redis 做分布式限流。
    """
    if request.url.path == "/agent/run":
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = _rate_window[ip]
        # 清掉 60 秒之前的记录
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= server_config.rate_limit_per_minute:
            raise HTTPException(
                status_code=429,
                detail=f"请求过于频繁，每分钟最多 {server_config.rate_limit_per_minute} 次",
            )
        window.append(now)

    return await call_next(request)


# ==================== 请求/响应模型 ====================

class AgentRequest(BaseModel):
    message: str = Field(..., description="用户输入，如『帮我找便宜的考研书』")
    session_id: str = Field("default", description="会话ID，用于多轮对话记忆")
    verbose: bool = Field(False, description="是否返回详细推理过程")


class ToolCallInfo(BaseModel):
    step: int
    tool: str
    level: str = Field(..., description="工具权限级别：safe / sensitive / dangerous")
    args: Dict[str, Any]
    result: str
    ok: bool = Field(True, description="该次工具调用是否成功")
    elapsed: float = Field(0.0, description="该次调用耗时（秒）")


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AgentResponse(BaseModel):
    answer: str = Field(..., description="Agent 的最终回答")
    steps: int = Field(..., description="推理步数")
    tool_calls: List[ToolCallInfo] = Field(default_factory=list, description="工具调用记录")
    elapsed: float = Field(..., description="总耗时（秒）")
    stopped_by: str = Field(..., description="停止原因：completed / max_steps / timeout / token_budget / llm_error")
    usage: UsageInfo = Field(default_factory=UsageInfo, description="Token 消耗")
    history_turns: int = Field(0, description="本轮之前的历史对话轮数（>0 说明记忆生效）")


# ==================== 接口 ====================

@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok" if _agent else "error",
        "agent_ready": _agent is not None,
        "error": _init_error,
        "tools": [fn.__name__ for fn in ALL_TOOLS],
        "limits": {
            "max_steps": agent_config.max_steps,
            "total_timeout": agent_config.total_timeout,
            "token_budget": agent_config.token_budget,
        },
    }


@app.get("/agent/tools")
async def list_tools():
    """查看 Agent 可用的工具及权限级别"""
    return {
        "count": len(ALL_TOOLS),
        "tools": [
            {
                "name": fn.__name__,
                "level": get_tool_level(fn.__name__).value,
                "description": fn.__doc__.strip().splitlines()[0],
            }
            for fn in ALL_TOOLS
        ],
        "description": get_tools_description(),
    }


@app.get("/metrics")
async def metrics():
    """运行统计"""
    total = _metrics["total_requests"]
    return {
        "total_requests": total,
        "total_tokens": _metrics["total_tokens"],
        "avg_tokens_per_request": round(_metrics["total_tokens"] / total, 1) if total else 0,
        "avg_elapsed": round(_metrics["total_elapsed"] / total, 2) if total else 0,
        "stopped_by": dict(_metrics["stopped_by"]),
        "memory": memory.stats(),
    }


@app.delete("/agent/session/{session_id}")
async def clear_session(session_id: str):
    """清空某个会话的对话记忆"""
    existed = memory.clear(session_id)
    return {"cleared": existed, "session_id": session_id}


@app.get("/agent/session/{session_id}")
async def get_session(session_id: str):
    """查看某个会话的历史（调试用）"""
    history = memory.get(session_id)
    return {
        "session_id": session_id,
        "turns": sum(1 for m in history if isinstance(m, HumanMessage)),
        "messages": [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant",
             "content": m.content[:200]}
            for m in history
        ],
    }


@app.post("/agent/run", response_model=AgentResponse)
async def run_agent(req: AgentRequest) -> AgentResponse:
    """
    执行 Agent

    完整走一遍 ReAct 循环：思考 → 调工具 → 观察结果 → 再思考 → ... → 回答
    受三重熔断保护：最大步数 / 总超时 / token 预算
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail=f"Agent 未就绪：{_init_error}")

    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message 不能为空")

    result = _agent.run(message, verbose=req.verbose, session_id=req.session_id)

    # 累计统计
    _metrics["total_requests"] += 1
    _metrics["total_tokens"] += result["usage"]["total_tokens"]
    _metrics["total_elapsed"] += result["elapsed"]
    _metrics["stopped_by"][result["stopped_by"]] += 1

    tool_calls = [
        ToolCallInfo(
            step=t["step"], tool=t["tool"], level=t.get("level", "safe"),
            args=t["args"], result=t["result"],
            ok=t.get("ok", True), elapsed=t.get("elapsed", 0.0),
        )
        for t in result["trace"]
        if t.get("type") == "tool_call"
    ]

    return AgentResponse(
        answer=result["answer"],
        steps=result["steps"],
        tool_calls=tool_calls,
        elapsed=result["elapsed"],
        stopped_by=result["stopped_by"],
        usage=UsageInfo(**result["usage"]),
        history_turns=result.get("history_turns", 0),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=server_config.port)
