"""
Agent 核心：Function Calling + ReAct 循环。

`run()` 是主循环：把 system prompt、历史对话和本轮提问拼成 messages，
反复调模型。模型返回 tool_calls 就执行工具，结果用 ToolMessage 回传
（必须带 tool_call_id，否则模型对不上是哪次请求的结果），
直到模型不再调工具为止。

循环里做了步数、总耗时、token 三个上限，防止它失控。
"""

import json
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from config import agent_config, llm_config
from memory import memory
from tools import TOOL_MAP, get_tool_level, ToolLevel


# ==================== System Prompt ====================

SYSTEM_PROMPT = """你是「小助手」，一个校园二手交易平台的智能购物助手。

## 你能做什么

你可以调用工具来帮用户完成任务，而不是凭空回答：

- **search_products** —— 搜索商品。用户想买东西时用。
- **get_product_detail** —— 查看商品详细信息和卖家。锁定商品后用。
- **get_seller_credit** —— 查询卖家信誉。用户担心安全，或你要评估风险时用。
- **search_faq** —— 查询平台规则（怎么发布、怎么取消订单等）。**问规则用这个，找商品用 search_products。**
- **create_order** —— 下单。**只有用户明确说要买才能调用。**

## 工作方式

1. 先理解用户的真实需求（一句话里可能有多个诉求）
2. 想清楚要调哪些工具、按什么顺序
3. 拿到结果后判断：信息够了吗？还需要查什么？
4. 信息齐全后，给用户清晰的答复

## 重要原则

- **不要编造数据**。价格、卖家、商品信息必须以工具返回的为准。
- **多步任务要拆解**。比如"帮我找便宜的考研书，看看卖家靠不靠谱"，
  需要先搜商品、再看详情、再查信誉，最后综合判断。
- **下单必须谨慎**。用户没说"买"，就只推荐、不下单。
- **用最少的步骤完成任务**。能一步查到的不要查三步，避免浪费用户时间。
- **用户提到具体商品时主动查**。比如"这个 iPad 多少钱""自行车怎么样"，
  应该直接用 search_products 搜一下给出结果，而不是反问"你说的是哪个"。
  只有用户描述极其模糊（如"那个东西"）时才需要澄清。
- 如果工具返回错误，尝试换参数，或如实告诉用户。
- 回答用简洁的中文，条理清晰，不要长篇大论。

## 输出格式

正常对话即可。回答可以带表情让语气友好些，但信息要准确。
"""


# ==================== 工具 Schema（Function Calling 协议） ====================

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "搜索校园二手平台的商品。当用户想买东西、找商品、"
                "问『有没有XX』时使用。商品涵盖教材、数码、运动器材等分类。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string",
                                "description": "搜索关键词，如『考研』『自行车』『教材』。留空表示不限。"},
                    "max_price": {"type": "number",
                                  "description": "最高价格（元）。用户说『便宜的』或给了预算时传入。"},
                    "category": {"type": "string",
                                 "description": "分类，可选：电子产品、书籍教材、生活用品、服饰鞋帽、运动器材、数码配件、文具办公、其他。"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_detail",
            "description": "查看某个商品的详细信息，包括描述和卖家。锁定商品后使用，通常跟在 search_products 之后。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "integer", "description": "商品 ID，从 search_products 结果里获取。"}
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_seller_credit",
            "description": "查询卖家信誉（历史成交次数、平均评分）。用户担心安全、问『卖家靠谱吗』时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "seller_id": {"type": "integer", "description": "卖家 ID，从 get_product_detail 结果里获取。"}
                },
                "required": ["seller_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_faq",
            "description": (
                "查询平台的使用规则和常见问题（基于知识库检索）。"
                "用户问『怎么发布商品』『订单怎么取消』这类**平台规则、操作说明**时使用。"
                "注意：这是查规则，不是查商品。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "用户的问题，尽量用原话。"}
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_order",
            "description": (
                "创建订单完成购买。**只有当用户明确表示要买"
                "（说『我要买』『帮我下单』）时才能调用**，不要自作主张。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "integer", "description": "要购买的商品 ID。"},
                    "buyer_id": {"type": "integer", "description": "买家 ID，默认 1（演示用）。"},
                },
                "required": ["product_id"],
            },
        },
    },
]


# ==================== Agent ====================

class CampusShoppingAgent:
    """校园购物助手 Agent"""

    def __init__(self) -> None:
        llm_config.validate()
        self.llm = ChatOpenAI(
            model=llm_config.model,
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            timeout=llm_config.request_timeout,
        ).bind_tools(TOOL_SCHEMAS)

    # ---------- 权限校验 ----------

    def _check_permission(self, tool_name: str) -> Optional[str]:
        """
        工具执行前的权限校验。

        危险工具（有写操作的）在生产环境应该走人工确认，
        这里先用配置开关兜住。
        """
        level = get_tool_level(tool_name)
        if level == ToolLevel.DANGEROUS and not agent_config.allow_dangerous_tools:
            return f"权限拒绝：工具 {tool_name} 属于危险操作，当前配置不允许执行。"
        return None

    # ---------- 工具执行 ----------

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行工具，带权限校验和退避重试。

        返回 {"result": str, "ok": bool, "elapsed": float}。
        不往外抛异常：抛异常会中断整个 Agent 循环，
        返回错误文本让模型自己判断（换参数还是如实告诉用户）更好。
        """
        start = time.time()

        # 权限校验
        deny = self._check_permission(name)
        if deny:
            return {"result": deny, "ok": False, "elapsed": 0.0}

        fn = TOOL_MAP.get(name)
        if fn is None:
            return {"result": f"错误：工具 {name} 不存在。", "ok": False, "elapsed": 0.0}

        last_err = None
        for attempt in range(agent_config.tool_max_retry + 1):
            try:
                result = fn(**args)
                return {"result": str(result), "ok": True,
                        "elapsed": round(time.time() - start, 2)}
            except TypeError as exc:
                # 参数错误：重试没意义，直接返回让模型修正参数
                return {"result": f"参数错误：{exc}。请检查参数格式后重试。",
                        "ok": False, "elapsed": round(time.time() - start, 2)}
            except Exception as exc:
                last_err = exc
                if attempt < agent_config.tool_max_retry:
                    # 退避重试：等待时间随重试次数递增，避免雪崩
                    time.sleep(agent_config.tool_retry_backoff * (attempt + 1))
                    continue

        return {"result": f"工具 {name} 执行失败（已重试 {agent_config.tool_max_retry} 次）：{last_err}",
                "ok": False, "elapsed": round(time.time() - start, 2)}

    # ---------- ReAct 循环 ----------

    def run(self, user_input: str, verbose: bool = True,
            session_id: str = "default") -> Dict[str, Any]:
        """
        执行一次完整的 Agent 推理

        三重熔断保护：
            ① max_steps     步数上限（防死循环）
            ② total_timeout 总耗时上限（防单步慢拖垮体验）
            ③ token_budget  累计 token 上限（防烧钱）

        :param session_id: 会话 ID，用于多轮对话记忆。
                           同一个 ID 的多次调用会带上历史上下文。

        Returns:
            {
              "answer":      最终回答
              "steps":       推理步数
              "trace":       每步详情（工具名/参数/结果/耗时）
              "elapsed":     总耗时
              "stopped_by":  停止原因
              "usage":       token 消耗统计
              "history_turns": 本轮之前的历史轮数（用于确认记忆生效）
            }
        """
        start = time.time()

        # 取该会话的历史对话（不含 SystemMessage）
        history = memory.get(session_id)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            *history,                        # ← 历史上下文拼在这里
            HumanMessage(content=user_input),
        ]
        history_turns = sum(1 for m in history if isinstance(m, HumanMessage))

        if verbose and history_turns:
            print(f"[记忆] 载入 {history_turns} 轮历史对话")
        trace: List[Dict[str, Any]] = []
        step = 0
        stopped_by = "completed"
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        while step < agent_config.max_steps:
            # ---- 熔断检查 1：总超时 ----
            if time.time() - start > agent_config.total_timeout:
                stopped_by = "timeout"
                if verbose:
                    print(f"[熔断] 总耗时超过 {agent_config.total_timeout}s，停止")
                break

            # ---- 熔断检查 2：token 预算 ----
            if usage["total_tokens"] > agent_config.token_budget:
                stopped_by = "token_budget"
                if verbose:
                    print(f"[熔断] Token 消耗超过预算 {agent_config.token_budget}，停止")
                break

            step += 1

            # ---- 第 1 步：模型思考，决定是否调工具 ----
            try:
                ai_msg: AIMessage = self.llm.invoke(messages)
            except Exception as exc:
                stopped_by = "llm_error"
                trace.append({"step": step, "type": "error", "content": str(exc)})
                return {
                    "answer": f"抱歉，调用大模型失败：{exc}",
                    "steps": step, "trace": trace,
                    "elapsed": round(time.time() - start, 2),
                    "stopped_by": stopped_by, "usage": usage,
                }

            # 累计 token（LangChain 把用量放在 response_metadata 里）
            meta = getattr(ai_msg, "response_metadata", {}) or {}
            tok = meta.get("token_usage") or meta.get("usage") or {}
            usage["prompt_tokens"] += tok.get("prompt_tokens", 0)
            usage["completion_tokens"] += tok.get("completion_tokens", 0)
            usage["total_tokens"] += tok.get("total_tokens", 0)

            messages.append(ai_msg)
            tool_calls = getattr(ai_msg, "tool_calls", None) or []

            # ---- 没有工具调用 → 模型认为可以回答了 ----
            if not tool_calls:
                stopped_by = "completed"
                if verbose:
                    print(f"[第{step}步] 模型给出最终答案")
                break

            # ---- 第 2 步：执行模型要求的所有工具 ----
            for tc in tool_calls:
                name = tc["name"]
                args = tc.get("args", {}) or {}

                if verbose:
                    print(f"[第{step}步] 调用工具：{name}({json.dumps(args, ensure_ascii=False)})")

                exec_result = self._execute_tool(name, args)
                result = exec_result["result"]

                if verbose:
                    preview = result.replace("\n", " ")[:80]
                    print(f"        结果：{preview}{'...' if len(result) > 80 else ''}")

                trace.append({
                    "step": step,
                    "type": "tool_call",
                    "tool": name,
                    "level": get_tool_level(name).value,
                    "args": args,
                    "result": result,
                    "ok": exec_result["ok"],
                    "elapsed": exec_result["elapsed"],
                })

                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        else:
            # while 正常跑完（没 break）说明达到最大步数
            stopped_by = "max_steps"
            if verbose:
                print(f"[熔断] 达到最大步数 {agent_config.max_steps}，强制停止")

        # ---- 取最后一条有内容的 AI 消息作为回答 ----
        answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                answer = msg.content
                break

        if not answer:
            answer = "抱歉，我没能完成任务。请换个说法再试试。"

        # 熔断时的补充说明
        if stopped_by == "max_steps":
            answer += "\n\n（提示：这个问题需要的步骤较多，我暂时只能给到以上信息。）"
        elif stopped_by == "timeout":
            answer += "\n\n（提示：处理超时了，可以先参考以上信息。）"

        # ---- 保存本轮对话到记忆 ----
        # 只存「用户消息 + 最终回答」，不存中间的 tool_call / tool_result。
        # 原因：工具消息必须成对出现（AIMessage.tool_calls ↔ ToolMessage），
        # 截断时容易破坏配对导致报错；而对话连续性靠最终回答就够，
        # 因为回答里通常已经包含了关键信息（商品名、价格等）。
        memory.save(session_id, [
            *history,
            HumanMessage(content=user_input),
            AIMessage(content=answer),
        ])

        return {
            "answer": answer,
            "steps": step,
            "history_turns": history_turns,
            "trace": trace,
            "elapsed": round(time.time() - start, 2),
            "stopped_by": stopped_by,
            "usage": usage,
        }


# ==================== 单例 ====================

_agent: Optional[CampusShoppingAgent] = None


def get_agent() -> CampusShoppingAgent:
    """获取 Agent 单例（模型初始化有开销，进程内复用）"""
    global _agent
    if _agent is None:
        _agent = CampusShoppingAgent()
    return _agent


# ==================== 命令行调试 ====================

if __name__ == "__main__":
    import sys

    agent = get_agent()

    if len(sys.argv) > 1:
        # 单次提问：每次用独立 session，不带历史
        question = " ".join(sys.argv[1:])
        print(f"\n用户：{question}\n")
        result = agent.run(question, session_id="cli_single")
        print(f"\n助手：{result['answer']}")
        print(f"\n[统计] {result['steps']} 步 / {result['elapsed']}s / "
              f"工具调用 {len(result['trace'])} 次 / "
              f"Token {result['usage']['total_tokens']} / "
              f"停止原因 {result['stopped_by']}")
    else:
        # 交互模式：同一个 session，支持多轮对话
        print("校园购物助手 Agent（输入 q 退出）")
        print("试试多轮对话，验证记忆是否生效：")
        print("  1) 帮我找便宜的考研书")
        print("  2) 那第二本多少钱          ← 考验上下文记忆")
        print("  3) 我要买它                ← 考验能否理解指代\n")

        session = "cli_interactive"
        while True:
            try:
                q = input("用户：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in ("q", "quit", "exit"):
                break
            if q.lower() == "clear":
                memory.clear(session)
                print("[已清空对话记忆]\n")
                continue

            res = agent.run(q, session_id=session)
            print(f"\n助手：{res['answer']}")
            print(f"\n[统计] {res['steps']} 步 / {res['elapsed']}s / "
                  f"工具 {len(res['trace'])} 次 / "
                  f"Token {res['usage']['total_tokens']} / "
                  f"历史 {res['history_turns']} 轮\n" + "-" * 60 + "\n")
