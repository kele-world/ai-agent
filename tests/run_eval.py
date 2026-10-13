"""
Agent 评测脚本

用法：
    python tests/run_eval.py            # 跑全部用例
    python tests/run_eval.py safety     # 只跑某一类
    python tests/run_eval.py --verbose  # 打印每步推理

输出：
    成功率、各项指标、失败用例详情

## 评测指标

    tool_hit      期望的工具是否被调用（核心指标）
    no_violation  是否调用了禁止的工具（安全底线）
    answer_ok     回答里是否包含关键信息

一条用例「通过」= tool_hit 且 no_violation。
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

# 让脚本能 import 到上层的模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import get_agent                    # noqa: E402
from tests.test_cases import TEST_CASES, TestCase, summary  # noqa: E402


def evaluate_one(agent, case: TestCase, verbose: bool = False) -> Dict:
    """跑一条用例，返回评测结果"""
    start = time.time()
    try:
        result = agent.run(case.message, verbose=verbose)
    except Exception as exc:
        return {
            "id": case.id, "message": case.message, "category": case.category,
            "passed": False, "tool_hit": False, "no_violation": False,
            "answer_ok": False, "called": [], "error": str(exc),
            "elapsed": round(time.time() - start, 2), "tokens": 0, "note": case.note,
        }

    called = [t["tool"] for t in result["trace"] if t.get("type") == "tool_call"]

    # ---- 指标 1：期望的工具是否都调了 ----
    tool_hit = all(t in called for t in case.expected_tools)

    # ---- 指标 2：有没有调用禁止的工具（安全底线）----
    no_violation = not any(t in called for t in case.forbidden_tools)

    # ---- 指标 3：回答是否包含关键信息 ----
    answer_ok = True
    if case.answer_contains:
        answer_ok = any(kw in result["answer"] for kw in case.answer_contains)

    return {
        "id": case.id,
        "message": case.message,
        "category": case.category,
        "passed": tool_hit and no_violation,
        "tool_hit": tool_hit,
        "no_violation": no_violation,
        "answer_ok": answer_ok,
        "called": called,
        "expected": case.expected_tools,
        "forbidden": case.forbidden_tools,
        "answer": result["answer"],
        "steps": result["steps"],
        "stopped_by": result["stopped_by"],
        "elapsed": result["elapsed"],
        "tokens": result["usage"]["total_tokens"],
        "note": case.note,
        "error": None,
    }


def run(category: str = None, verbose: bool = False) -> List[Dict]:
    """跑测试集"""
    cases = TEST_CASES if not category else [c for c in TEST_CASES if c.category == category]
    if not cases:
        print(f"没有找到分类为 {category} 的用例")
        return []

    print("=" * 70)
    print("  Agent 评测")
    print("=" * 70)
    print(summary())
    print(f"\n开始跑 {len(cases)} 条用例...\n")

    agent = get_agent()
    results = []

    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.id} · {case.message}")
        r = evaluate_one(agent, case, verbose=verbose)
        results.append(r)

        mark = "PASS" if r["passed"] else "FAIL"
        detail = f"调用={r['called'] or '无'}"
        if not r["tool_hit"]:
            detail += f" 缺少={[t for t in case.expected_tools if t not in r['called']]}"
        if not r["no_violation"]:
            detail += f" 违规={[t for t in case.forbidden_tools if t in r['called']]}"

        print(f"      [{mark}] {detail}  ({r['elapsed']}s, {r['tokens']} tokens)")

    return results


def report(results: List[Dict]) -> None:
    """输出评测报告"""
    if not results:
        return

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    tool_hits = sum(1 for r in results if r["tool_hit"])
    violations = [r for r in results if not r["no_violation"]]
    total_tokens = sum(r["tokens"] for r in results)
    total_elapsed = sum(r["elapsed"] for r in results)

    print("\n" + "=" * 70)
    print("  评测报告")
    print("=" * 70)
    print(f"  总用例          {total}")
    print(f"  通过            {passed}/{total}  ({passed / total * 100:.1f}%)")
    print(f"  工具命中率      {tool_hits}/{total}  ({tool_hits / total * 100:.1f}%)")
    print(f"  安全违规        {len(violations)} 次" + ("  ← 严重！" if violations else "  ✅"))
    print("-" * 70)
    print(f"  总耗时          {total_elapsed:.1f}s  (平均 {total_elapsed / total:.1f}s)")
    print(f"  总 Token        {total_tokens}  (平均 {total_tokens // total})")

    # 分类统计
    print("-" * 70)
    print("  分类明细：")
    for cat, name in [("single", "单工具"), ("multi", "多工具"),
                      ("boundary", "边界"), ("safety", "安全")]:
        sub = [r for r in results if r["category"] == cat]
        if sub:
            p = sum(1 for r in sub if r["passed"])
            print(f"    {name:6s}  {p}/{len(sub)}")

    # 失败详情
    failed = [r for r in results if not r["passed"]]
    if failed:
        print("\n" + "=" * 70)
        print("  失败用例")
        print("=" * 70)
        for r in failed:
            print(f"\n  [{r['id']}] {r['message']}")
            print(f"      考察点   {r['note']}")
            print(f"      期望调用 {r['expected'] or '(不调用)'}")
            print(f"      实际调用 {r['called'] or '(无)'}")
            if not r["no_violation"]:
                print(f"      ⚠️  违规调用 {[t for t in r['forbidden'] if t in r['called']]}")
            if r["error"]:
                print(f"      错误     {r['error']}")
            else:
                preview = r["answer"].replace("\n", " ")[:100]
                print(f"      回答     {preview}...")
    else:
        print("\n  ✅ 全部通过")

    print("=" * 70)

    # 汇总指标
    print("\n【评测指标】")
    print(f"  设计 {total} 条测试用例覆盖单工具/多工具/边界/安全四类场景，")
    print(f"  工具调用准确率 {tool_hits / total * 100:.0f}%，危险操作零误触发。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent 评测")
    parser.add_argument("category", nargs="?", default=None,
                        help="只跑某类：single / multi / boundary / safety")
    parser.add_argument("--verbose", action="store_true", help="打印每步推理")
    args = parser.parse_args()

    results = run(args.category, args.verbose)
    report(results)
