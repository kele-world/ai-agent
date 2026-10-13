"""
配置中心。

之前配置散在 agent.py / tools.py / main.py 里到处 os.getenv，
改一个参数要在几个文件里找。集中到一处之后，改配置只动这里，
启动时也能一次性校验（Key 没填直接报错，而不是跑到一半才崩）。

分层：
    LLMConfig     大模型相关
    AgentConfig   Agent 行为控制（熔断、超时、预算）
    ServerConfig  HTTP 服务
    DBConfig      数据源
    RedisConfig   会话记忆
    RagConfig     search_faq 依赖的 RAG 服务
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() in ("1", "true", "yes")


# ==================== 大模型 ====================

@dataclass
class LLMConfig:
    api_key: str = field(default_factory=lambda: _env("DEEPSEEK_API_KEY"))
    base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "https://api.deepseek.com/v1"))
    model: str = field(default_factory=lambda: _env("LLM_MODEL", "deepseek-chat"))
    temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.3))
    max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 1000))
    # 单次 LLM 调用的超时（秒）
    request_timeout: int = field(default_factory=lambda: _env_int("LLM_REQUEST_TIMEOUT", 60))

    def validate(self) -> None:
        """启动时校验，避免跑到一半才发现 Key 没配"""
        if not self.api_key or self.api_key.startswith("sk-xxx"):
            raise RuntimeError(
                "未配置 DEEPSEEK_API_KEY。\n"
                "请复制 .env.example 为 .env 并填入真实 Key。"
            )


# ==================== Agent 行为 ====================

@dataclass
class AgentConfig:
    # ---- 熔断 ----
    # 最大推理步数：防止模型反复调同一个工具陷入死循环
    max_steps: int = field(default_factory=lambda: _env_int("AGENT_MAX_STEPS", 8))

    # 整个任务的总超时（秒）：即使步数没超，时间到了也强制停止。
    # 单步慢（网络差）时 8 步可能要跑好几分钟，用户体验太差
    total_timeout: int = field(default_factory=lambda: _env_int("AGENT_TOTAL_TIMEOUT", 180))

    # ---- 成本控制 ----
    # Token 预算：累计消耗超过就停止（防止单次任务烧太多钱）
    token_budget: int = field(default_factory=lambda: _env_int("AGENT_TOKEN_BUDGET", 30000))

    # ---- 工具执行 ----
    # 工具失败重试次数
    tool_max_retry: int = field(default_factory=lambda: _env_int("TOOL_MAX_RETRY", 2))
    # 重试退避基数（秒），实际等待 = base * (attempt+1)
    tool_retry_backoff: float = field(default_factory=lambda: _env_float("TOOL_RETRY_BACKOFF", 0.5))
    # 单个工具的执行超时（秒）
    tool_timeout: int = field(default_factory=lambda: _env_int("TOOL_TIMEOUT", 15))

    # ---- 安全 ----
    # 是否允许执行危险工具（如下单）。生产环境应设为 false，改由人工确认
    allow_dangerous_tools: bool = field(default_factory=lambda: _env_bool("ALLOW_DANGEROUS_TOOLS", True))


# ==================== HTTP 服务 ====================

@dataclass
class ServerConfig:
    port: int = field(default_factory=lambda: _env_int("PORT", 8100))
    # 单个 IP 每分钟最大请求数（简单限流，防止被刷）
    rate_limit_per_minute: int = field(default_factory=lambda: _env_int("RATE_LIMIT_PER_MINUTE", 30))


# ==================== 数据源 ====================

@dataclass
class DBConfig:
    # 是否连接真实 MySQL。false 则用内置模拟数据，方便零依赖运行
    use_real_db: bool = field(default_factory=lambda: _env_bool("USE_REAL_DB", False))
    host: str = field(default_factory=lambda: _env("DB_HOST", "localhost"))
    port: int = field(default_factory=lambda: _env_int("DB_PORT", 3306))
    user: str = field(default_factory=lambda: _env("DB_USER", "root"))
    password: str = field(default_factory=lambda: _env("DB_PASSWORD", ""))
    database: str = field(default_factory=lambda: _env("DB_NAME", "esshop"))


# ==================== 会话记忆（Redis） ====================

@dataclass
class RedisConfig:
    """
    会话记忆存储配置

    用 Redis 而不是进程内存：内存版重启就丢、多实例不共享、
    还得自己写淘汰逻辑；Redis 天然有 TTL 清理。
    """
    enabled: bool = field(default_factory=lambda: _env_bool("REDIS_ENABLED", True))
    host: str = field(default_factory=lambda: _env("REDIS_HOST", "localhost"))
    port: int = field(default_factory=lambda: _env_int("REDIS_PORT", 6379))
    password: str = field(default_factory=lambda: _env("REDIS_PASSWORD", ""))
    db: int = field(default_factory=lambda: _env_int("REDIS_DB", 0))
    # key 前缀：避免和 esshop 项目的登录 token 冲突
    prefix: str = field(default_factory=lambda: _env("REDIS_PREFIX", "agent:memory:"))
    # 会话过期时间（秒），每次对话刷新
    ttl: int = field(default_factory=lambda: _env_int("REDIS_TTL", 1800))
    # 连接超时（毫秒）
    timeout: int = field(default_factory=lambda: _env_int("REDIS_TIMEOUT", 5000))


# ==================== RAG 服务（用于 search_faq 工具） ====================

@dataclass
class RagConfig:
    # RAG 服务地址（另一个项目的 ai-service）
    # 不可用时会自动降级到 MySQL 关键词匹配
    service_url: str = field(default_factory=lambda: _env("RAG_SERVICE_URL", "http://localhost:8000"))
    enabled: bool = field(default_factory=lambda: _env_bool("RAG_ENABLED", True))
    timeout: int = field(default_factory=lambda: _env_int("RAG_TIMEOUT", 60))


# ==================== 全局单例 ====================

llm_config = LLMConfig()
agent_config = AgentConfig()
server_config = ServerConfig()
db_config = DBConfig()
rag_config = RagConfig()
redis_config = RedisConfig()


def print_summary() -> None:
    """启动时打印配置摘要（Key 打码），方便排查环境问题"""
    masked = (llm_config.api_key[:8] + "***") if llm_config.api_key else "(未配置)"
    print("=" * 56)
    print("  配置摘要")
    print("=" * 56)
    print(f"  模型        {llm_config.model}")
    print(f"  接口        {llm_config.base_url}")
    print(f"  Key         {masked}")
    print(f"  数据源      {'MySQL(真实)' if db_config.use_real_db else '内置模拟数据'}")
    print(f"  RAG 服务    {rag_config.service_url if rag_config.enabled else '已禁用'}")
    print(f"  记忆存储    {redis_config.host}:{redis_config.port} (TTL {redis_config.ttl}s)"
          if redis_config.enabled else "  记忆存储    内存（Redis 已禁用）")
    print("-" * 56)
    print(f"  最大步数    {agent_config.max_steps}")
    print(f"  总超时      {agent_config.total_timeout}s")
    print(f"  Token 预算  {agent_config.token_budget}")
    print(f"  危险工具    {'允许' if agent_config.allow_dangerous_tools else '禁止'}")
    print("=" * 56)
