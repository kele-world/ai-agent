"""
Agent 工具集。

模型靠工具的 description 判断什么时候调它，描述写得含糊模型就会
漏调或乱调，所以每个工具的描述都把使用场景和边界写清楚了。

权限分级：查商品这类只读操作随便调，下单这种有副作用的标为
DANGEROUS，Agent 执行前校验（见 agent._check_permission）。

工具异常不往外抛，在 agent._execute_tool 里消化掉返回错误文本，
让模型自己决定重试、换参数还是如实告知用户。

工具清单：
    search_products     搜索商品           SAFE
    get_product_detail  商品详情           SAFE
    get_seller_credit   卖家信誉           SAFE
    search_faq          平台知识问答(RAG)  SAFE
    create_order        下单               DANGEROUS
"""

import os
from enum import Enum
from typing import Optional

from config import db_config, rag_config


# ==================== 工具权限分级 ====================

class ToolLevel(str, Enum):
    """
    工具权限级别

    SAFE      只读，无副作用，可自由调用
    SENSITIVE 涉及隐私或需要额外校验
    DANGEROUS 有写操作，生产环境应要求人工确认
    """
    SAFE = "safe"
    SENSITIVE = "sensitive"
    DANGEROUS = "dangerous"


# 工具名 → 权限级别
TOOL_LEVELS = {
    "search_products": ToolLevel.SAFE,
    "get_product_detail": ToolLevel.SAFE,
    "get_seller_credit": ToolLevel.SAFE,
    "search_faq": ToolLevel.SAFE,
    "create_order": ToolLevel.DANGEROUS,
}


# ==================== 数据层 ====================

# 内置模拟数据：USE_REAL_DB=false 时使用，让项目零依赖即可运行
MOCK_PRODUCTS = [
    {"id": 1, "title": "捷安特ATX山地车 27速", "price": 850.0, "category": "运动器材",
     "seller_id": 2, "seller_name": "ww01",
     "description": "骑了两年，车况良好，刚换过刹车片和链条。适合通勤和周末骑行。"},
    {"id": 2, "title": "考研英语一 历年真题解析", "price": 45.0, "category": "书籍教材",
     "seller_id": 3, "seller_name": "ww02",
     "description": "张剑黄皮书，2020-2024年，无笔记无划线，几乎全新。"},
    {"id": 3, "title": "高数教材 同济第七版 上下册", "price": 25.0, "category": "书籍教材",
     "seller_id": 3, "seller_name": "ww02",
     "description": "有少量笔记，不影响阅读。两本一起出。"},
    {"id": 4, "title": "iPad Air 4 64G WiFi版", "price": 2100.0, "category": "电子产品",
     "seller_id": 2, "seller_name": "ww01",
     "description": "2022年买的，贴膜带壳，屏幕无划痕，电池健康度92%。"},
    {"id": 5, "title": "小米移动电源 20000mAh", "price": 65.0, "category": "数码配件",
     "seller_id": 2, "seller_name": "ww01",
     "description": "买了没用几次，支持快充，可带上飞机。"},
    {"id": 6, "title": "耐克Air Force 1 白色 42码", "price": 420.0, "category": "服饰鞋帽",
     "seller_id": 3, "seller_name": "ww02",
     "description": "穿过三四次，鞋底干净，无开胶。原盒还在。"},
    {"id": 7, "title": "宜家书桌 120x60cm", "price": 120.0, "category": "生活用品",
     "seller_id": 2, "seller_name": "ww01",
     "description": "毕业出，桌面无划痕。需自提，宿舍楼下交易。"},
    {"id": 8, "title": "罗技MX Master 3 无线鼠标", "price": 380.0, "category": "数码配件",
     "seller_id": 2, "seller_name": "ww01",
     "description": "95新，仅拆封试用。支持多设备切换，办公神器。"},
]

MOCK_SELLERS = {
    2: {"name": "ww01", "trade_count": 12, "rating": 4.8, "join_days": 320},
    3: {"name": "ww02", "trade_count": 5, "rating": 4.5, "join_days": 150},
}

# 模拟 FAQ：RAG 服务不可用时的降级数据
MOCK_FAQ = [
    {"q": "怎么发布商品", "a": "登录后点击右上角「发布」按钮，填写标题、分类、售价和描述，上传实物图（最多6张），点击发布即可。"},
    {"q": "怎么取消订单", "a": "进入「我的订单」，找到对应订单点击「取消」。待付款和已付款都可以取消，取消后商品会自动重新上架。"},
    {"q": "订单状态有哪些", "a": "共4种：待付款（下单成功）、已付款（等待线下交易）、已完成（交易结束）、已取消。"},
    {"q": "怎么改价", "a": "进入「我的发布」页面，找到商品点击「改价」，输入新价格即可。"},
    {"q": "交易安全注意什么", "a": "建议在校内公共场所当面交易，验货后再付款，贵重物品建议结伴。不要提前转账给陌生人。"},
    {"q": "怎么收藏商品", "a": "在商品详情页点击「收藏」按钮，之后可以在头像菜单的「我的收藏」里查看。"},
    {"q": "商品下架了还能看到吗", "a": "下架后其他人在商品市场看不到，但你在「我的发布」里仍能看到，可以重新上架。"},
    {"q": "怎么修改个人信息", "a": "点击右上角头像 → 个人中心，可以修改头像、昵称和手机号。"},
]


def _query_db(sql: str, params: tuple = ()):
    """查询 MySQL（仅 USE_REAL_DB=true 时启用）"""
    import pymysql
    conn = pymysql.connect(
        host=db_config.host, port=db_config.port,
        user=db_config.user, password=db_config.password,
        database=db_config.database, charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()


def _execute_db(sql: str, params: tuple = ()) -> int:
    """执行写操作，返回影响行数"""
    import pymysql
    conn = pymysql.connect(
        host=db_config.host, port=db_config.port,
        user=db_config.user, password=db_config.password,
        database=db_config.database, charset="utf8mb4",
    )
    try:
        with conn.cursor() as cur:
            n = cur.execute(sql, params)
        conn.commit()
        return n
    finally:
        conn.close()


# ==================== 工具 1：搜索商品 ====================

def search_products(keyword: str = "", max_price: Optional[float] = None,
                    category: str = "") -> str:
    """
    搜索校园二手平台的商品。

    当用户想买东西、找商品、询问"有没有XX"时使用这个工具。

    Args:
        keyword: 搜索关键词，比如"考研""自行车""教材"。留空表示不限关键词。
        max_price: 最高价格（元）。用户说"便宜的""预算XX以内"时传入。
        category: 商品分类，可选值：电子产品、书籍教材、生活用品、
                  服饰鞋帽、运动器材、数码配件、文具办公、其他。

    Returns:
        匹配的商品列表（含ID、标题、价格、分类）；没有匹配时返回提示语。
    """
    try:
        if db_config.use_real_db:
            sql = ("SELECT id, title, price FROM product "
                   "WHERE deleted=0 AND status='ON_SALE'")
            params = []
            if keyword:
                sql += " AND title LIKE %s"
                params.append(f"%{keyword}%")
            if max_price is not None:
                sql += " AND price <= %s"
                params.append(max_price)
            rows = _query_db(sql, tuple(params))
            items = [{"id": r["id"], "title": r["title"], "price": float(r["price"])}
                     for r in rows]
        else:
            items = [
                {"id": p["id"], "title": p["title"], "price": p["price"],
                 "category": p["category"]}
                for p in MOCK_PRODUCTS
                if (not keyword or keyword in p["title"] or keyword in p["description"])
                and (max_price is None or p["price"] <= max_price)
                and (not category or category == p["category"])
            ]

        if not items:
            return "没有找到符合条件的商品。可以试试放宽价格范围或换个关键词。"

        lines = [f"找到 {len(items)} 件商品："]
        for it in items[:10]:
            cat = f" [{it['category']}]" if it.get("category") else ""
            lines.append(f"  - ID={it['id']} 《{it['title']}》 ¥{it['price']}{cat}")
        return "\n".join(lines)

    except Exception as exc:
        return f"搜索商品失败：{exc}"


# ==================== 工具 2：商品详情 ====================

def get_product_detail(product_id: int) -> str:
    """
    查看某个商品的详细信息，包括描述和卖家信息。

    当用户已锁定某件商品、想看更多细节（成色、描述、谁在卖）时使用。
    通常跟在 search_products 之后。

    Args:
        product_id: 商品 ID，从 search_products 的结果里获取。

    Returns:
        商品的详细信息；商品不存在时返回提示语。
    """
    try:
        if db_config.use_real_db:
            rows = _query_db(
                "SELECT p.id, p.title, p.price, p.description, p.user_id, u.nickname "
                "FROM product p LEFT JOIN user u ON p.user_id = u.id "
                "WHERE p.id = %s AND p.deleted = 0", (product_id,))
            if not rows:
                return f"商品 ID={product_id} 不存在。"
            r = rows[0]
            return (f"商品详情：\n"
                    f"  标题：{r['title']}\n"
                    f"  价格：¥{r['price']}\n"
                    f"  描述：{r['description'] or '暂无描述'}\n"
                    f"  卖家：{r['nickname']}（ID={r['user_id']}）")
        else:
            p = next((x for x in MOCK_PRODUCTS if x["id"] == product_id), None)
            if not p:
                return f"商品 ID={product_id} 不存在。"
            return (f"商品详情：\n"
                    f"  标题：{p['title']}\n"
                    f"  价格：¥{p['price']}\n"
                    f"  分类：{p['category']}\n"
                    f"  描述：{p['description']}\n"
                    f"  卖家：{p['seller_name']}（ID={p['seller_id']}）")

    except Exception as exc:
        return f"查询商品详情失败：{exc}"


# ==================== 工具 3：卖家信誉 ====================

def get_seller_credit(seller_id: int) -> str:
    """
    查询卖家的信誉情况（历史成交次数、平均评分）。

    当用户担心交易安全、问"这个卖家靠谱吗"时使用。
    也可以在帮用户做购买决策前主动调用，评估交易风险。

    Args:
        seller_id: 卖家用户 ID，从 get_product_detail 的结果里获取。

    Returns:
        卖家的信誉数据；卖家不存在时返回提示语。
    """
    try:
        if db_config.use_real_db:
            rows = _query_db(
                "SELECT u.id, u.nickname, "
                "(SELECT COUNT(*) FROM order_info o WHERE o.seller_id=u.id AND o.status='COMPLETED') AS trade_count, "
                "(SELECT ROUND(AVG(rating),1) FROM product_review r WHERE r.seller_id=u.id AND r.deleted=0) AS rating "
                "FROM user u WHERE u.id = %s", (seller_id,))
            if not rows:
                return f"卖家 ID={seller_id} 不存在。"
            r = rows[0]
            rating = r["rating"] if r["rating"] is not None else "暂无评分"
            return (f"卖家信誉：\n"
                    f"  昵称：{r['nickname']}\n"
                    f"  历史成交：{r['trade_count']} 次\n"
                    f"  平均评分：{rating}")
        else:
            s = MOCK_SELLERS.get(seller_id)
            if not s:
                return f"卖家 ID={seller_id} 不存在。"
            level = ("信誉良好" if s["rating"] >= 4.5 and s["trade_count"] >= 5
                     else "交易记录较少，建议当面验货")
            return (f"卖家信誉：\n"
                    f"  昵称：{s['name']}\n"
                    f"  历史成交：{s['trade_count']} 次\n"
                    f"  平均评分：{s['rating']} / 5.0\n"
                    f"  注册天数：{s['join_days']} 天\n"
                    f"  评估：{level}")

    except Exception as exc:
        return f"查询卖家信誉失败：{exc}"


# ==================== 工具 4：平台知识问答（RAG） ====================

def search_faq(question: str) -> str:
    """
    查询平台的使用规则和常见问题（基于 RAG 检索）。

    当用户问"怎么发布商品""订单怎么取消""平台怎么用"这类
    **平台规则、操作说明**相关的问题时使用。
    注意：这个工具查的是平台规则，不是商品。找商品请用 search_products。

    Args:
        question: 用户的问题，尽量用原话，比如"怎么取消订单"。

    Returns:
        检索到的平台知识；服务不可用时降级到关键词匹配。
    """
    # 优先走 RAG 服务（另一个项目的 ai-service）
    if rag_config.enabled:
        try:
            import requests
            resp = requests.post(
                f"{rag_config.service_url}/chat",
                json={"session_id": "agent_faq", "message": question},
                timeout=rag_config.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            # RAG 服务返回 degraded=true 说明它自己也降级了
            if not data.get("degraded") and data.get("reply"):
                return f"平台知识：\n{data['reply']}"
        except Exception:
            # RAG 服务不可用是常态（没启动/超时），静默降级到下面的关键词匹配
            pass

    # 降级方案：关键词匹配（保证 RAG 挂了也能回答）
    try:
        if db_config.use_real_db:
            rows = _query_db(
                "SELECT question, answer, keywords FROM chat_knowledge "
                "WHERE deleted=0 AND enabled=1")
            faqs = [{"q": r["question"], "a": r["answer"],
                     "kw": (r.get("keywords") or "")} for r in rows]
        else:
            faqs = [{"q": f["q"], "a": f["a"], "kw": f["q"]} for f in MOCK_FAQ]

        # 简单打分：命中字符越多越相关
        best, best_score = None, 0
        for f in faqs:
            text = f["q"] + f.get("kw", "")
            score = sum(1 for ch in set(question) if ch in text)
            if score > best_score:
                best, best_score = f, score

        if best and best_score >= 2:
            return f"平台知识：\n【{best['q']}】\n{best['a']}"

        return "没有找到相关的平台说明。可以换个说法，或联系平台管理员。"
    except Exception as exc:
        return f"查询平台知识失败：{exc}"


# ==================== 工具 5：下单（危险操作） ====================

def create_order(product_id: int, buyer_id: int = 1) -> str:
    """
    创建订单完成购买。

    **只有当用户明确表示要买（说"我要买""帮我下单"）时才能调用。**
    不要自作主张下单 —— 这是有实际副作用的操作。

    Args:
        product_id: 要购买的商品 ID。
        buyer_id: 买家用户 ID，默认 1（演示用）。

    Returns:
        下单结果，包含订单号；失败时返回原因。
    """
    try:
        if db_config.use_real_db:
            rows = _query_db(
                "SELECT id, title, price, user_id, status FROM product "
                "WHERE id=%s AND deleted=0", (product_id,))
            if not rows:
                return f"下单失败：商品 ID={product_id} 不存在。"
            p = rows[0]
            if p["status"] != "ON_SALE":
                return f"下单失败：《{p['title']}》当前状态为 {p['status']}，无法购买。"
            if p["user_id"] == buyer_id:
                return "下单失败：不能购买自己发布的商品。"

            import uuid
            from datetime import datetime
            order_no = "ES" + datetime.now().strftime("%Y%m%d%H%M%S") + uuid.uuid4().hex[:6]
            _execute_db(
                "INSERT INTO order_info (order_no, product_id, buyer_id, seller_id, amount, status) "
                "VALUES (%s,%s,%s,%s,%s,'PENDING_PAYMENT')",
                (order_no, product_id, buyer_id, p["user_id"], p["price"]))
            _execute_db("UPDATE product SET status='SOLD' WHERE id=%s", (product_id,))
            return (f"下单成功！\n"
                    f"  订单号：{order_no}\n"
                    f"  商品：{p['title']}\n"
                    f"  金额：¥{p['price']}\n"
                    f"  状态：待付款")
        else:
            p = next((x for x in MOCK_PRODUCTS if x["id"] == product_id), None)
            if not p:
                return f"下单失败：商品 ID={product_id} 不存在。"
            return (f"下单成功（模拟）！\n"
                    f"  商品：{p['title']}\n"
                    f"  金额：¥{p['price']}\n"
                    f"  卖家：{p['seller_name']}\n"
                    f"  状态：待付款\n"
                    f"  提示：请与卖家约定校内当面交易")

    except Exception as exc:
        return f"下单失败：{exc}"


# ==================== 工具注册表 ====================

ALL_TOOLS = [
    search_products,
    get_product_detail,
    get_seller_credit,
    search_faq,
    create_order,
]

TOOL_MAP = {fn.__name__: fn for fn in ALL_TOOLS}


def get_tool_level(name: str) -> ToolLevel:
    """获取工具的权限级别"""
    return TOOL_LEVELS.get(name, ToolLevel.SENSITIVE)


def get_tools_description() -> str:
    """工具清单（给人和模型看）"""
    lines = []
    for fn in ALL_TOOLS:
        level = get_tool_level(fn.__name__).value
        summary = fn.__doc__.strip().splitlines()[0]
        lines.append(f"- {fn.__name__} [{level}]: {summary}")
    return "\n".join(lines)
