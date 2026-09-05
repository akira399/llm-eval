"""版本回归对比（方案 §7）：把两次运行的记录并排比较。

例：
  python scripts/summarize.py runs/bm25-baseline-*.jsonl runs/topk8-*.jsonl
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalkit.runner import JUDGE_DIMENSIONS  # noqa: E402

DIM_NAMES = {
    "correctness_mean": "正确性",
    "faithfulness_mean": "忠实度",
    "format_mean": "格式",
    "tone_mean": "语气",
}


def load_run(pattern: str) -> list[dict]:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"找不到运行记录：{pattern}")
    path = matches[-1]  # 同版本多次运行取最近一次
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def aggregate(records: list[dict]) -> tuple[dict, dict]:
    """返回（总体统计, 分类目统计）。"""
    def agg(subset: list[dict]) -> dict:
        import statistics

        out = {"n": len(subset)}
        out["n_rejected"] = sum(1 for r in subset if r["target"]["rejected"])
        for dim in JUDGE_DIMENSIONS:
            scores = [
                r["judge"][dim]["score"]
                for r in subset
                if r.get("judge") and r["judge"].get(dim) and r["judge"][dim].get("score") is not None
            ]
            if scores:
                out[f"{dim}_mean"] = round(statistics.mean(scores), 3)
        return out

    categories = sorted({r["case"]["category"] for r in records})
    return agg(records), {cat: agg([r for r in records if r["case"]["category"] == cat]) for cat in categories}


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    runs = [(arg, load_run(arg)) for arg in sys.argv[1:3]]
    overall = {}
    for label, records in runs:
        overall[label], _ = aggregate(records)

    print("===== 总体对比 =====")
    keys = ["n", "n_rejected"] + [f"{d}_mean" for d in JUDGE_DIMENSIONS]
    header = f"{'指标':<12}" + "".join(f"{os.path.basename(label):>24}" for label, _ in runs)
    print(header)
    for key in keys:
        name = DIM_NAMES.get(key, key)
        row = f"{name:<12}"
        values = [overall[label].get(key, "-") for label, _ in runs]
        row += "".join(f"{v:>24}" for v in values)
        if len(values) == 2 and isinstance(values[0], (int, float)) and isinstance(values[1], (int, float)):
            delta = values[1] - values[0]
            row += f"    Δ {delta:+.3f}" if isinstance(delta, float) else f"    Δ {delta:+d}"
        print(row)

    print("\n===== 分类目（正确性均分）=====")
    cats = sorted({c for _, rec in runs for c in {x["case"]["category"] for x in rec}})
    for cat in cats:
        row = f"{cat:<12}"
        for label, records in runs:
            _, by_cat = aggregate(records)
            row += f"{by_cat.get(cat, {}).get('correctness_mean', '-'):>24}"
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
