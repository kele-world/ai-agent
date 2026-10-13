"""数据库连接自检 —— 切到 USE_REAL_DB=true 后用来验证"""

import sys
from pathlib import Path

# 让脚本能 import 到上层模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import db_config, print_summary  # noqa: E402

print_summary()
print()
print("数据源模式:", "真实 MySQL" if db_config.use_real_db else "内置模拟数据")
print(f"连接信息: {db_config.user}@{db_config.host}:{db_config.port}/{db_config.database}")
print()

if not db_config.use_real_db:
    print("当前是模拟数据模式，跳过数据库检查。")
    raise SystemExit(0)

from tools import search_products, get_product_detail, get_seller_credit, search_faq

print("=" * 56)
print("  真实数据库查询测试")
print("=" * 56)

print("\n[1] 搜索商品（关键词=教材）")
print(search_products(keyword="教材"))

print("\n[2] 商品详情（ID=3）")
print(get_product_detail(product_id=3))

print("\n[3] 卖家信誉（ID=3）")
print(get_seller_credit(seller_id=3))

print("\n[4] 平台知识（RAG 未启动时应降级到关键词匹配）")
print(search_faq(question="怎么取消订单"))

print("\n" + "=" * 56)
print("  检查完成")
print("=" * 56)
