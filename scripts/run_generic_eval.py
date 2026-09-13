"""通用评测入口（Phase 0）：任意 TargetAdapter + JudgeProfile 跑任意套件。

与 scripts/run_eval.py（RAG 回归基线）并列；泛化后的执行路径：
  suites/*.yaml（generic）或 cases/*.yaml（RAG 兼容转换）
    → registry.build_target(target_id)
    → EvaluationEngine + JudgeProfile
例：
  python scripts/run_generic_eval.py --list-targets
  python scripts/run_generic_eval.py --suite suites/demo-chat.yaml --target demo-chat --version demo1
  python scripts/run_generic_eval.py --suite cases/poke-rag-v0.yaml --target pokerag-local --limit 3
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalkit.contracts import evaluation_case_from_rag, load_evaluation_cases  # noqa: E402
from evalkit.engine import EvaluationEngine  # noqa: E402
from evalkit.judge_profile import (  # noqa: E402
    RagJudgeProfile,
    chat_profile,
    json_profile,
    none_profile,
)
from evalkit.registry import TARGET_CATALOG, build_target  # noqa: E402
from evalkit.schema import load_cases  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_suite(path: str):
    """按 meta.format 分派加载：generic 直读；RAG 套件转换为通用用例。"""
    meta, cases = load_cases(path) if _is_rag(path) else load_evaluation_cases(path)
    if _is_rag(path):
        cases = [evaluation_case_from_rag(c) for c in cases]
    return meta, cases


def _is_rag(path: str) -> bool:
    import yaml

    with open(path, encoding="utf-8") as f:
        meta = (yaml.safe_load(f) or {}).get("meta") or {}
    return meta.get("format") != "generic"


def pick_profile(target_id: str, use_llm: bool):
    kind = TARGET_CATALOG.get(target_id, {}).get("kind")
    if kind == "chat":
        return chat_profile(use_llm=use_llm)
    if kind == "json":
        return json_profile()
    if kind == "rag":
        return RagJudgeProfile()
    return none_profile()


def main() -> int:
    parser = argparse.ArgumentParser(description="通用评测（任意适配器 × 任意套件）")
    parser.add_argument("--suite", help="套件文件（generic 或 RAG 格式均可）")
    parser.add_argument("--target", default="demo-chat", help=f"目标 id：{sorted(TARGET_CATALOG)}")
    parser.add_argument("--version", default="generic-run")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--use-llm", action="store_true", help="chat 类启用 LLM 语气/相关性评分")
    parser.add_argument("--list-targets", action="store_true")
    args = parser.parse_args()

    if args.list_targets:
        for tid, info in TARGET_CATALOG.items():
            print(f"{tid:<14} {info['kind']:<6} {info['description']}")
        return 0
    if not args.suite:
        parser.error("需要 --suite（或用 --list-targets 查看目标）")

    meta, cases = load_suite(args.suite)
    engine = EvaluationEngine(adapter=build_target(args.target), profile=pick_profile(args.target, args.use_llm))
    summary = engine.run(cases, version=args.version, limit=args.limit)

    print(f"\n===== 通用评测完成 · {args.version} =====")
    print(f"套件 {meta.get('name', '?')} · 目标 {args.target} · "
          f"用例 {summary['n_cases']} · 拒答 {summary['n_rejected']} · 错误 {summary['n_errors']} · "
          f"耗时 {summary['latency_ms_avg']}ms")
    if summary["judge_means"]:
        print("维度均分：" + " · ".join(f"{d} {v}" for d, v in summary["judge_means"].items()))
    print(f"记录：runs/{summary['run_file']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
