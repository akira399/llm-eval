"""运行一次评测（入口脚本，用法见 docs/00-技术方案.md §9）。

例：
  python scripts/run_eval.py --limit 3 --no-judge          # 冒烟：3 条，不做 LLM 裁判
  python scripts/run_eval.py --version bm25-baseline       # 全量 50 条 + 四维裁判
  python scripts/run_eval.py --override '{"rag": {"top_k_answer": 8}}' --version topk8
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalkit.judge import LLMJudge          # noqa: E402
from evalkit.providers.pokerag import PokeRagProvider  # noqa: E402
from evalkit.runner import Runner           # noqa: E402
from evalkit.schema import load_cases       # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-Eval 运行一次评测")
    parser.add_argument("--cases", default=os.path.join(_ROOT, "cases", "poke-rag-v0.yaml"))
    parser.add_argument("--version", default="baseline", help="版本标签（回归对比的 key）")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条（冒烟用）")
    parser.add_argument("--workers", type=int, default=1, help="并发数（默认 1 串行）")
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM 裁判（只跑被测应用+规则格式分）")
    parser.add_argument("--mode", choices=["inprocess", "http"], default="inprocess")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765", help="http 模式的被测服务地址")
    parser.add_argument("--poke-rag-root", default=None)
    parser.add_argument("--override", default=None, help='被测系统配置覆盖，JSON，如 \'{"rag": {"top_k_answer": 8}}\'')
    parser.add_argument("--out", default=os.path.join(_ROOT, "runs"))
    args = parser.parse_args()

    meta, cases = load_cases(args.cases)
    overrides = json.loads(args.override) if args.override else None
    provider = PokeRagProvider(
        root=args.poke_rag_root, overrides=overrides, mode=args.mode, base_url=args.base_url
    )
    judge = None if args.no_judge else LLMJudge()

    version = args.version
    if overrides:
        version = f"{version}"  # 版本标签由调用者负责区分；生效配置已记录在 summary
    runner = Runner(provider=provider, judge=judge, out_dir=args.out)
    summary = runner.run(cases, version=version, workers=args.workers, limit=args.limit)

    print(f"\n===== 评测完成 · {version} =====")
    print(f"用例 {summary['n_cases']} · 作答 {summary['n_answered']} · "
          f"拒答 {summary['n_rejected']} · 错误 {summary['n_errors']} · "
          f"平均耗时 {summary['latency_ms_avg']}ms")
    if summary["judge_means"]:
        print("四维均分：" + " · ".join(
            f"{dim} {score}" for dim, score in summary["judge_means"].items()
        ))
    print("\n分类目：")
    print(f"{'类目':<10}{'数量':>4}{'拒答':>4}{'错误':>4}"
          + ("".join(f"{dim:>14}" for dim in ("correctness_mean", "faithfulness_mean", "format_mean", "tone_mean"))
             if summary["by_category"] and any("correctness_mean" in v for v in summary["by_category"].values()) else ""))
    for cat, stat in summary["by_category"].items():
        row = f"{cat:<10}{stat['n']:>4}{stat['n_rejected']:>4}{stat['n_errors']:>4}"
        for dim in ("correctness_mean", "faithfulness_mean", "format_mean", "tone_mean"):
            row += f"{stat.get(dim, '-'):>14}"
        print(row)
    print(f"\n记录：runs/{summary['run_file']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
