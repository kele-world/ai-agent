# 校园购物助手 Agent

这是一个基于 Function Calling + ReAct 的多步推理 Agent，接在校园二手交易平台上做购物助手。

用户的一句话里经常有多个诉求，比如说：

> 帮我找两本便宜的考研书，顺便看看卖家靠不靠谱，下单后怎么取消？

这句话要拆成三步：搜商品、查卖家信誉、查平台取消规则，然后综合回答。写 if-else 很难覆盖所有组合，所以交给模型自己决定调哪些工具、调几次、按什么顺序——它调完一个拿到结果，再决定下一步，直到信息够了给答案。

## 跑起来

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

copy .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
```

默认 `USE_REAL_DB=false`，用内置模拟数据，不装 MySQL 也能跑。

命令行交互：

```bash
python agent.py
```

单次提问：

```bash
python agent.py "帮我找便宜的考研书，卖家靠谱吗"
```

起 HTTP 服务 + 前端页面：

```bash
python main.py
# 然后直接打开 index.html
```

跑评测：

```bash
python tests/run_eval.py
python tests/run_eval.py safety     # 只跑安全类
```

Docker 部署（含 Redis）：

```bash
start-docker.bat
# 或
docker compose up -d
```

## 工具

| 工具 | 作用 | 权限 |
|------|------|------|
| `search_products` | 按关键词/价格/分类搜商品 | safe |
| `get_product_detail` | 商品详情和卖家 | safe |
| `get_seller_credit` | 卖家信誉（成交次数、评分） | safe |
| `search_faq` | 平台规则问答，走 RAG 检索 | safe |
| `create_order` | 下单 | dangerous |

`create_order` 有副作用，执行前会过一遍权限校验。`ALLOW_DANGEROUS_TOOLS=false` 可以直接禁掉，线上建议这么配，改成人工确认。

## 几个实现上的取舍

**熔断。** Agent 最常见的失控方式是反复调同一个工具，或者一次任务烧掉太多 token。所以循环里加了三道检查：步数上限（默认 8）、总耗时上限（180s）、token 预算（30000）。任意一道触发就停下，并在回答后面补一句说明，而不是直接报错——告诉用户"我能给到这些，但可能不完整"比甩一个失败好。

**工具异常不往外抛。** 工具抛异常会直接打断整个推理链。所以异常在 `_execute_tool` 内部消化掉，返回错误文本让模型自己判断：是换个参数重试，还是如实告诉用户查不到。另外参数错误（`TypeError`）不做重试，因为同样的参数再试也没意义，直接让模型改参数。

**记忆只存问答对。** 存 Redis，key 是 `agent:memory:{session_id}`，默认 TTL 30 分钟，每次对话刷新。只存用户消息和最终回答，不存中间的 tool_call / tool_result——工具消息必须成对出现（AIMessage.tool_calls 和 ToolMessage 一一对应），截断历史时很容易把配对切散导致接口报错。而对话连续性靠最终回答就够了，回答里一般已经带了商品名和价格。历史按"轮"截断而不是按条，一条用户消息到下一条用户消息算一轮。

Redis 挂了会自动退回进程内存实现——记忆是增强功能，不该因为它挂掉让整个 Agent 崩了。但内存版重启就丢、多实例不共享，只适合单机演示。

**配置集中在一处。** `config.py` 把散落的 `os.getenv` 收拢成几个 dataclass，启动时统一校验（Key 没配就直接报错，不会跑到一半才崩）。

## 评测

Agent 的输出不确定，同一个问题模型可能换关键词、换工具顺序，所以评测断言的是行为而不是字面回答：该调的工具调了吗、不该调的调了吗、关键信息答出来了吗。

`tests/test_cases.py` 里有 11 条用例，分四类——单工具（考察不过度调用）、多工具（考察任务拆解）、边界（考察兜底，查不到时如实说）、安全（考察危险工具会不会误触发）。其中安全类最要紧，Agent 有副作用，乱下单比答错话严重得多。

## 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查，含配置摘要 |
| GET | `/agent/tools` | 工具列表和权限级别 |
| POST | `/agent/run` | 执行 Agent，带 `session_id` 就自动带上历史 |
| GET | `/agent/session/{id}` | 看某会话的历史（调试用） |
| DELETE | `/agent/session/{id}` | 清空某会话记忆 |
| GET | `/metrics` | 累计请求数、token 消耗、记忆后端类型 |

`/agent/run` 带 `session_id` 时支持多轮，比如上一轮搜到商品、这一轮说"25 块那本我要了"能接上。单 IP 每分钟限 30 次，因为每次调用都烧 token。

## 文件

```
config.py      配置中心（分层 + 启动校验）
memory.py      会话记忆（Redis，自动降级内存）
tools.py       5 个工具 + 权限分级
agent.py       Agent 核心：ReAct 循环、熔断、记忆
main.py        FastAPI 服务（限流、监控）
index.html     前端，把推理过程可视化
tests/         测试用例、评测脚本、数据库自检
Dockerfile / docker-compose.yml
```

## 后续想做的

- 多 Agent 协作（搜索、比价、议价分开）
- 长期记忆，记住用户偏好
- 一次返回多个 tool_call 时并行执行（现在是串行）
- 接 LangSmith 看每步的 token 和延迟# ai-agent
