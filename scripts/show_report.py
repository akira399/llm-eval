"""评测报表（M1）：可视化查看任意一次/两次运行的结果。

启动（在 llm-eval 目录下）：
  ../poke-rag/.venv/Scripts/python.exe -m streamlit run scripts/show_report.py --server.port=8502
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from evalkit.agreement import annotations_path, compute_agreement, load_annotations
from evalkit.runner import summarize

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(_ROOT, "runs")
DIM_NAMES = {"correctness": "正确性", "faithfulness": "引用忠实度", "format": "格式", "tone": "语气"}


def list_runs() -> list[str]:
    if not os.path.isdir(RUNS_DIR):
        return []
    return sorted(
        (f for f in os.listdir(RUNS_DIR) if f.endswith(".jsonl")),
        key=lambda f: os.path.getmtime(os.path.join(RUNS_DIR, f)),
        reverse=True,
    )


def load_run(name: str) -> list[dict]:
    with open(os.path.join(RUNS_DIR, name), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dim_means(records: list[dict]) -> dict[str, float]:
    out = {}
    for dim in DIM_NAMES:
        scores = [r["judge"][dim]["score"] for r in records
                  if r.get("judge") and r["judge"].get(dim) and r["judge"][dim].get("score") is not None]
        if scores:
            out[dim] = sum(scores) / len(scores)
    return out


def category_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    cats = sorted({r["case"]["category"] for r in records})
    for cat in cats:
        subset = [r for r in records if r["case"]["category"] == cat]
        row = {"类目": cat, "数量": len(subset),
               "拒答": sum(1 for r in subset if r["target"]["rejected"]),
               "错误": sum(1 for r in subset if r["target"]["error"])}
        for dim, mean in dim_means(subset).items():
            row[DIM_NAMES[dim]] = round(mean, 3)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="LLM-Eval 评测报表", page_icon="📊", layout="wide")
    st.title("📊 LLM-Eval 评测报表")

    runs = list_runs()
    if not runs:
        st.error("runs/ 里没有评测记录。先运行 scripts/run_eval.py。")
        return

    with st.sidebar:
        primary = st.selectbox("主运行记录", runs, format_func=lambda f: f.replace(".jsonl", ""))
        compare = st.selectbox("对比运行记录（可选）", ["（不对比）"] + runs,
                               format_func=lambda f: f if f == "（不对比）" else f.replace(".jsonl", ""))

    records = load_run(primary)
    st.caption(f"共 {len(records)} 条 · 文件 runs/{primary}")

    tab_overview, tab_cat, tab_cmp, tab_fail, tab_agree = st.tabs(
        ["总览", "分类目", "版本对比", "失败清单", "裁判一致性"])

    with tab_overview:
        answered = [r for r in records if not r["target"]["rejected"] and not r["target"]["error"]]
        cols = st.columns(5)
        cols[0].metric("用例数", len(records))
        cols[1].metric("正常作答", len(answered))
        cols[2].metric("拒答", sum(1 for r in records if r["target"]["rejected"]))
        cols[3].metric("运行错误", sum(1 for r in records if r["target"]["error"]))
        lat = [r["target"]["latency_ms"] for r in answered]
        cols[4].metric("平均耗时", f"{int(sum(lat) / len(lat))}ms" if lat else "-")
        means = dim_means(records)
        if means:
            cols2 = st.columns(len(means))
            for col, (dim, mean) in zip(cols2, means.items()):
                col.metric(DIM_NAMES[dim], round(mean, 3))
        diff = {"easy": [r for r in records if r["case"]["difficulty"] == "easy"],
                "hard": [r for r in records if r["case"]["difficulty"] == "hard"]}
        if diff["hard"]:
            st.markdown("**已知弱项（hard）单独看：**")
            for name, subset in diff.items():
                m = dim_means(subset)
                st.caption(f"{name}: {len(subset)} 条 · " + " · ".join(
                    f"{DIM_NAMES[d]} {round(v, 2)}" for d, v in m.items()))

    with tab_cat:
        st.dataframe(category_table(records), use_container_width=True, hide_index=True)

    with tab_cmp:
        if compare == "（不对比）":
            st.info("在左侧选择第二份运行记录即可对比（版本回归）。")
        else:
            other = load_run(compare)
            left, right = summarize(records, version=primary.replace(".jsonl", "")), \
                summarize(other, version=compare.replace(".jsonl", ""))
            rows = []
            for dim in DIM_NAMES:
                a, b = left["judge_means"].get(dim), right["judge_means"].get(dim)
                if a is not None and b is not None:
                    rows.append({"维度": DIM_NAMES[dim], left["version"]: a,
                                 right["version"]: b, "Δ": round(b - a, 3)})
            for key, name in (("n_rejected", "拒答数"), ("latency_ms_avg", "平均耗时(ms)")):
                a, b = left.get(key), right.get(key)
                if a is not None and b is not None:
                    rows.append({"维度": name, left["version"]: a,
                                 right["version"]: b, "Δ": round(b - a, 2)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab_fail:
        fails = [r for r in records
                 if r["target"]["error"]
                 or (r.get("judge") and (r["judge"].get("correctness") or {}).get("score") is not None
                     and r["judge"]["correctness"]["score"] < 0.5)
                 or (r.get("judge") and (r["judge"].get("faithfulness") or {}).get("score") is not None
                     and r["judge"]["faithfulness"]["score"] < 0.5)]
        st.markdown(f"正确性或忠实度低于 0.5 / 运行出错的用例：**{len(fails)} 条**")
        for r in fails:
            with st.expander(f"{r['case']['id']} · {r['case']['category']} · {r['case']['query']}"):
                st.markdown(f"**回答：** {r['target']['answer_text'] or '（出错：' + r['target']['error'] + '）'}")
                judge = r.get("judge") or {}
                for dim in DIM_NAMES:
                    d = judge.get(dim) or {}
                    if d.get("score") is not None:
                        st.caption(f"{DIM_NAMES[dim]}={d['score']} · {d.get('reason', '')}"
                                   + (f" · 未支持断言：{d['unsupported_claims']}" if d.get("unsupported_claims") else ""))

    with tab_agree:
        ann_file = annotations_path(os.path.join(RUNS_DIR, primary))
        annotations = load_annotations(ann_file)
        if not annotations:
            st.info("这份运行还没有人工标注。启动标注界面：\n\n"
                    "```\n../poke-rag/.venv/Scripts/python.exe -m streamlit run scripts/annotate_app.py\n```")
        else:
            result = compute_agreement(records, annotations)
            st.markdown(f"已标注 **{result['n_annotated']}** 条（协议：先人工盲评，再比对 AI 裁判）")
            rows = []
            for dim, stat in result["dimensions"].items():
                rows.append({"维度": DIM_NAMES[dim], "可比对": stat.get("n_compared", 0),
                             "一致率": stat.get("agreement", "-"), "Kappa": stat.get("kappa", "-")})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("Kappa 判读：≥0.8 优秀 · ≥0.6 良好 · ≥0.4 中等 · <0.4 需要修订评分细则")
            dis = [(dim, d) for dim, stat in result["dimensions"].items()
                   for d in stat.get("disagreements", [])]
            if dis:
                st.markdown(f"**分歧样本 {len(dis)} 条：**")
                for dim, d in dis:
                    st.caption(f"{d['case_id']} · {DIM_NAMES[dim]}：人工={d['human']} / AI={d['ai']}"
                               + (f" · {d['notes']}" if d.get("notes") else ""))


if __name__ == "__main__":
    main()
