"""评测报告生成器：把一次运行 + 人工标注（如有）汇总成 Markdown 报告。

产物落在 reports/（入库），是《评测报告 PDF》的文字底稿。
用法：
  ../poke-rag/.venv/Scripts/python.exe scripts/make_report.py [runs/xxx.jsonl]
"""
import glob
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalkit.agreement import annotations_path, compute_agreement, load_annotations  # noqa: E402
from evalkit.runner import summarize  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM_NAMES = {"correctness": "正确性", "faithfulness": "引用忠实度", "format": "格式", "tone": "语气"}


def latest_run() -> str:
    matches = sorted(glob.glob(os.path.join(_ROOT, "runs", "*.jsonl")), key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError("runs/ 里没有评测记录")
    return matches[-1]


def md_table(headers: list[str], rows: list[list]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return "\n".join([head, sep, body])


def main() -> int:
    run_file = sys.argv[1] if len(sys.argv) > 1 else latest_run()
    if not os.path.isabs(run_file):
        run_file = os.path.join(_ROOT, run_file)
    with open(run_file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    # 版本名与生效配置优先读随行 summary（运行记录本身不携带运行元数据）
    summary_path = os.path.splitext(run_file)[0] + ".summary.json"
    version = os.path.splitext(os.path.basename(run_file))[0]
    cfg: dict = {}
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            saved = json.load(f)
        version = saved.get("version") or version
        cfg = saved.get("effective_config") or {}

    summary = summarize(records, version=version)
    lines = [
        f"# LLM 应用效果评测报告 · {version}",
        "",
        f"- 运行记录：`runs/{os.path.basename(run_file)}`（{records[0]['meta']['ts']}）",
        f"- 被测应用：Poke-RAG（模型 `{cfg.get('llm', {}).get('model', '?')}`，"
        f"BM25-only={'是' if not cfg.get('rag', {}).get('use_dense') else '否'}，"
        f"top_k_answer={cfg.get('rag', {}).get('top_k_answer', '?')}）",
        f"- 用例集：`cases/poke-rag-v0.yaml`（{summary['n_cases']} 条 · 10 类目）",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 1. 总体结果",
        "",
        md_table(
            ["指标", "数值"],
            [["正常作答", f"{summary['n_answered']}"],
             ["拒答", f"{summary['n_rejected']}"],
             ["运行错误", f"{summary['n_errors']}"],
             ["平均耗时", f"{summary['latency_ms_avg']} ms"],
             *[[DIM_NAMES[d], mean] for d, mean in summary["judge_means"].items()]],
        ),
        "",
        "## 2. 分类目结果",
        "",
        md_table(
            ["类目", "数量", "拒答", "错误", "正确性", "忠实度", "格式", "语气"],
            [[cat, stat["n"], stat["n_rejected"], stat["n_errors"],
              stat.get("correctness_mean", "-"), stat.get("faithfulness_mean", "-"),
              stat.get("format_mean", "-"), stat.get("tone_mean", "-")]
             for cat, stat in summary["by_category"].items()],
        ),
        "",
        "## 3. 失败与风险清单",
        "",
    ]
    fails = [r for r in records
             if r["target"]["error"]
             or ((r.get("judge") or {}).get("correctness") or {}).get("score") is not None
             and r["judge"]["correctness"]["score"] < 0.5]
    if not fails:
        lines.append("本轮无正确性低于 0.5 的用例。")
    else:
        for r in fails:
            judge = r.get("judge") or {}
            reason = ((judge.get("correctness") or {}).get("facts")
                      and [f["fact"] for f in judge["correctness"]["facts"] if not f.get("covered")])
            lines.append(f"- **{r['case']['id']}**（{r['case']['category']}）{r['case']['query']}"
                         + (f" —— 未覆盖要点：{reason}" if reason else
                            f" —— {r['target']['error'] or '正确性低于 0.5'}"))

    ann_path = annotations_path(run_file)
    annotations = load_annotations(ann_path)
    lines += ["", "## 4. AI 裁判可信性（人工盲评一致性）", ""]
    if not annotations:
        lines.append("> 待人工标注后运行 `scripts/agreement.py` 生成。标注是「AI 裁判可信」的核心证据。")
    else:
        agreement = compute_agreement(records, annotations)
        lines += [md_table(
            ["维度", "可比对", "一致率", "Kappa"],
            [[DIM_NAMES[dim], stat.get("n_compared", 0), stat.get("agreement", "-"),
              ("-（退化）" if stat.get("degenerate") else stat.get("kappa", "-"))]
             for dim, stat in agreement["dimensions"].items()],
        ), "",
        "Kappa 判读：≥0.8 优秀 · ≥0.6 良好 · ≥0.4 中等。标注「退化」表示至少一方"
        "全部同判（Kappa 数学上失去意义，以一致率为准）——忠实度/格式维度的人工标注"
        "全为同一档，恰恰说明被测应用没有出现编造或格式问题。"]

    out_dir = os.path.join(_ROOT, "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"评测报告-{version}-{datetime.now().strftime('%Y%m%d')}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"报告已生成：{out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
