"""
会话记忆管理。

默认存 Redis，key 是 `agent:memory:{session_id}`，TTL 默认 30 分钟，
每次对话刷新。Redis 不可用时自动退回进程内存实现。

只存用户消息和最终回答，不存中间的 tool_call / tool_result：
工具消息必须成对出现（AIMessage.tool_calls 和 ToolMessage 一一对应），
截断历史时容易把配对切散导致接口报错。而对话连续性靠最终回答就够，
回答里一般已经带了商品名和价格。

历史按「轮」截断（一条用户消息到下一条用户消息算一轮），
这样不会把一轮对话切成两半。序列化用 JSON，跨进程可读，
方便直接拿 redis-cli 看。
"""

import json
import threading
from collections import OrderedDict
from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from config import redis_config


# ==================== 序列化 ====================

def _to_storable(messages: List[BaseMessage]) -> str:
    """LangChain 消息 → JSON 字符串"""
    data = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant",
         "content": m.content}
        for m in messages
    ]
    return json.dumps(data, ensure_ascii=False)


def _from_storable(raw: str) -> List[BaseMessage]:
    """JSON 字符串 → LangChain 消息"""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    out: List[BaseMessage] = []
    for item in data:
        content = item.get("content", "")
        if item.get("role") == "user":
            out.append(HumanMessage(content=content))
        else:
            out.append(AIMessage(content=content))
    return out


# ==================== 内存实现（降级用） ====================

class _MemoryBackend:
    """进程内存实现 —— Redis 不可用时的兜底"""

    def __init__(self, max_sessions: int = 200) -> None:
        self.max_sessions = max_sessions
        self._sessions: "OrderedDict[str, List[BaseMessage]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, session_id: str) -> List[BaseMessage]:
        with self._lock:
            if session_id in self._sessions:
                self._sessions.move_to_end(session_id)
                return list(self._sessions[session_id])
            return []

    def save(self, session_id: str, messages: List[BaseMessage]) -> None:
        with self._lock:
            self._sessions[session_id] = list(messages)
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def clear_all(self) -> int:
        with self._lock:
            n = len(self._sessions)
            self._sessions.clear()
            return n

    def session_ids(self) -> List[str]:
        with self._lock:
            return list(self._sessions.keys())


# ==================== Redis 实现 ====================

class _RedisBackend:
    """Redis 实现 —— 支持多实例共享、自动过期"""

    def __init__(self) -> None:
        import redis  # 延迟导入，未安装时不影响内存模式

        self._client = redis.Redis(
            host=redis_config.host,
            port=redis_config.port,
            password=redis_config.password or None,
            db=redis_config.db,
            socket_timeout=redis_config.timeout / 1000.0,
            decode_responses=True,       # 自动解码成 str，省去手动 decode
        )
        self._prefix = redis_config.prefix
        self._ttl = redis_config.ttl
        # 连通性验证
        self._client.ping()

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    def get(self, session_id: str) -> List[BaseMessage]:
        raw = self._client.get(self._key(session_id))
        if not raw:
            return []
        # 访问即续期，避免正在聊的会话中途过期
        self._client.expire(self._key(session_id), self._ttl)
        return _from_storable(raw)

    def save(self, session_id: str, messages: List[BaseMessage]) -> None:
        self._client.setex(self._key(session_id), self._ttl, _to_storable(messages))

    def clear(self, session_id: str) -> bool:
        return self._client.delete(self._key(session_id)) > 0

    def clear_all(self) -> int:
        # 只删自己的 key，不碰 Redis 里的其他数据（比如 esshop 的登录 token）
        keys = list(self._client.scan_iter(match=f"{self._prefix}*", count=100))
        if not keys:
            return 0
        return self._client.delete(*keys)

    def session_ids(self) -> List[str]:
        ids = []
        for k in self._client.scan_iter(match=f"{self._prefix}*", count=100):
            ids.append(k[len(self._prefix):])
        return ids


# ==================== 统一入口 ====================

class MemoryStore:
    """
    会话记忆存储

    自动选择后端：Redis 可用则用 Redis，否则降级到内存。
    """

    def __init__(self, max_turns: int = 5) -> None:
        self.max_turns = max_turns
        self._backend = None
        self._backend_name = "none"
        self._init_backend()

    def _init_backend(self) -> None:
        if redis_config.enabled:
            try:
                self._backend = _RedisBackend()
                self._backend_name = "redis"
                print(f"[记忆] 使用 Redis 存储: {redis_config.host}:{redis_config.port}"
                      f" (TTL {redis_config.ttl}s)")
                return
            except Exception as exc:
                print(f"[记忆] Redis 不可用，降级到内存存储: {exc}")

        self._backend = _MemoryBackend()
        self._backend_name = "memory"

    @property
    def backend_name(self) -> str:
        return self._backend_name

    # ---------- 读 ----------

    def get(self, session_id: str) -> List[BaseMessage]:
        """取某会话的历史消息（不含 SystemMessage）"""
        try:
            return self._backend.get(session_id)
        except Exception as exc:
            # Redis 抖动时不能让对话失败，返回空历史继续
            print(f"[记忆] 读取失败，按无历史处理: {exc}")
            return []

    # ---------- 写 ----------

    def save(self, session_id: str, messages: List[BaseMessage]) -> None:
        """保存会话历史，自动截断到 max_turns 轮"""
        try:
            self._backend.save(session_id, self._trim(messages))
        except Exception as exc:
            print(f"[记忆] 保存失败（不影响本次对话）: {exc}")

    def clear(self, session_id: str) -> bool:
        try:
            return self._backend.clear(session_id)
        except Exception:
            return False

    def clear_all(self) -> int:
        try:
            return self._backend.clear_all()
        except Exception:
            return 0

    # ---------- 截断 ----------

    def _trim(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        按「轮」截断，保留最近 max_turns 轮

        一轮 = 从一条 HumanMessage 到下一条 HumanMessage 之前。
        按轮截断保证不会把一轮对话切一半。
        """
        human_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
        if len(human_indices) <= self.max_turns:
            return list(messages)
        start = human_indices[len(human_indices) - self.max_turns]
        return list(messages[start:])

    # ---------- 统计 ----------

    def stats(self) -> Dict:
        """记忆占用情况（供 /metrics 用）"""
        try:
            ids = self._backend.session_ids()
        except Exception:
            ids = []

        sample = {}
        for sid in ids[:10]:
            try:
                msgs = self._backend.get(sid)
                sample[sid] = sum(1 for m in msgs if isinstance(m, HumanMessage))
            except Exception:
                pass

        return {
            "backend": self._backend_name,
            "active_sessions": len(ids),
            "max_turns_per_session": self.max_turns,
            "sessions": sample,
        }


# 全局单例
memory = MemoryStore()
