"""人机一致性 CLI：算一致率 + Cohen's Kappa，打印并落盘 agreement 文件。

用法（标完一轮后运行）：
  ../poke-rag/.venv/Scripts/python.exe scripts/agreement.py [runs/xxx.jsonl]
不指定运行记录时用最新一份。
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalkit.agreement import annotations_path, compute_agreement, load_annotations  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM_NAMES = {"correctness": "正确性", "faithfulness": "引用忠实度", "format": "格式", "tone": "语气"}


def latest_run() -> str:
    matches = sorted(glob.glob(os.path.join(_ROOT, "runs", "*.jsonl")), key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError("runs/ 里没有评测记录，先运行 scripts/run_eval.py")
    return matches[-1]


def main() -> int:
    run_file = sys.argv[1] if len(sys.argv) > 1 else latest_run()
    if not os.path.isabs(run_file):
        run_file = os.path.join(_ROOT, run_file)
    with open(run_file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    ann_path = annotations_path(run_file)
    annotations = load_annotations(ann_path)
    if not annotations:
        print(f"还没有人工标注：{ann_path}")
        print("先运行标注界面： ../poke-rag/.venv/Scripts/python.exe -m streamlit run scripts/annotate_app.py")
        return 1

    result = compute_agreement(records, annotations)
    result["run"] = os.path.basename(run_file)

    print(f"\n===== 人机一致性 · {os.path.basename(run_file)} · 已标注 {result['n_annotated']} 条 =====")
    print(f"{'维度':<8}{'可比对':>6}{'一致率':>8}{'Kappa':>8}   判读")
    for dim, stat in result["dimensions"].items():
        if not stat.get("n_compared"):
            print(f"{DIM_NAMES[dim]:<8}{0:>6}     -       -    （无可比对样本）")
            continue
        kappa = stat["kappa"]
        verdict = ("优秀" if kappa and kappa >= 0.8 else
                   "良好" if kappa and kappa >= 0.6 else
                   "中等" if kappa and kappa >= 0.4 else "需要修订细则")
        print(f"{DIM_NAMES[dim]:<8}{stat['n_compared']:>6}{stat['agreement']:>8.1%}"
              f"{kappa if kappa is not None else '-':>8}   {verdict}")

    disagreements = [(dim, d) for dim, stat in result["dimensions"].items()
                     for d in stat.get("disagreements", [])]
    if disagreements:
        print(f"\n分歧样本（{len(disagreements)} 条）——修订评分细则的原料：")
        for dim, d in disagreements:
            print(f"  {d['case_id']} · {DIM_NAMES[dim]}：人工={d['human']} / AI={d['ai']}"
                  + (f"（AI 分 {d['ai_score']}）" if d.get("ai_score") is not None else "")
                  + (f" · 备注：{d['notes']}" if d.get("notes") else ""))

    out_path = ann_path.replace(".jsonl", ".agreement.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
