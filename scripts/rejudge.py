"""换裁判重算（评分与运行分离的价值）：不重跑被测应用，只重算四维评分。

用途：修订裁判提示词/换裁判模型后，对已有运行记录重算分数。
产物：{原文件名}.rejudged.jsonl + .summary.json（原始记录保留不动）。

用法：
  ../poke-rag/.venv/Scripts/python.exe scripts/rejudge.py [runs/xxx.jsonl]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalkit.judge import LLMJudge          # noqa: E402
from evalkit.runner import Runner, summarize  # noqa: E402
from evalkit.schema import Case, TargetResult  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM_NAMES = {"correctness": "正确性", "faithfulness": "引用忠实度", "format": "格式", "tone": "语气"}


def latest_run() -> str:
    import glob

    matches = sorted(glob.glob(os.path.join(_ROOT, "runs", "*.jsonl")), key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError("runs/ 里没有评测记录")
    return matches[-1]


def main() -> int:
    run_file = sys.argv[1] if len(sys.argv) > 1 else latest_run()
    if not os.path.isabs(run_file):
        run_file = os.path.join(_ROOT, run_file)
    with open(run_file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    judge = LLMJudge()
    for i, rec in enumerate(records, 1):
        case = Case.from_dict(rec["case"])
        target = TargetResult(**rec["target"])
        rec["judge"] = judge.judge(case, target)
        print(f"[{i}/{len(records)}] {case.id} 重新判分完成", flush=True)

    version = os.path.basename(run_file).rsplit("-", 2)[0]
    summary = summarize(records, version=version)
    out_path = os.path.splitext(run_file)[0] + ".rejudged.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary["run_file"] = os.path.basename(out_path)
    with open(out_path.replace(".jsonl", ".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n===== 重判完成 · {version} =====")
    print("四维均分：" + " · ".join(f"{DIM_NAMES[d]} {v}" for d, v in summary["judge_means"].items()))
    print(f"记录：runs/{summary['run_file']}（原记录保留）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
