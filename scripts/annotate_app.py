"""人工盲评标注工具（M1 核心）——给标注员（你）用的极简界面。

协议（docs/00 §5）：先人工盲评，再算一致率。界面默认**不显示** AI 裁判
的分数和理由，保证盲评独立。

启动（Git Bash，在 llm-eval 目录下）：
  ../poke-rag/.venv/Scripts/python.exe -m streamlit run scripts/annotate_app.py

浏览器会自动打开；每条答 3~4 个选择题，点「保存本条」自动跳到下一条
未标注的题。已保存的随时可以改，改完再点保存即可（同一条以最后一次为准）。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from evalkit.agreement import annotations_path, save_annotation

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(_ROOT, "runs")

ANSWER_QUESTIONS = [
    ("correctness", "① 正确性：回答覆盖了参考要点吗？", ("对", "部分对", "错"),
     "对照「参考要点」逐条看：全部覆盖=对；只答出一部分=部分对；答错或没答上=错"),
    ("faithfulness", "② 忠实度：有没有编造（说出引用片段里没有的内容）？", ("没有编造", "有编造嫌疑", "明显编造"),
     "展开「引用的知识片段」对照：数字/名称与片段一致=没有编造；拿不准=有嫌疑；片段里根本没有=明显编造"),
    ("format", "③ 格式：引用编号、长度、有没有乱码？", ("没问题", "有问题"),
     "答案里有 [1] 这样的编号、读起来通顺=没问题"),
    ("tone", "④ 语气：读起来舒服吗？", ("好", "一般", "差"),
     "简洁友好专业=好；啰嗦/生硬=一般；敷衍/答非所问的语气=差"),
]
REJECT_QUESTIONS = [
    ("correctness", "① 拒答行为：是否恰当地承认了「不知道/超出知识范围」且没有编造？", ("对", "错"),
     "明确说无法回答且没有硬编一个答案=对；编了答案=错"),
    ("format", "② 格式：简短（一般几句话）、通顺？", ("没问题", "有问题"), ""),
    ("tone", "③ 语气：礼貌、友好吗？", ("好", "一般", "差"), ""),
]


def list_runs() -> list[str]:
    if not os.path.isdir(RUNS_DIR):
        return []
    return sorted(
        (f for f in os.listdir(RUNS_DIR) if f.endswith(".jsonl")),
        key=lambda f: os.path.getmtime(os.path.join(RUNS_DIR, f)),
        reverse=True,
    )


def load_run(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    st.set_page_config(page_title="LLM-Eval 人工标注", page_icon="📝", layout="centered")
    st.title("📝 人工盲评标注")

    runs = list_runs()
    if not runs:
        st.error("runs/ 里没有评测记录。先运行 scripts/run_eval.py 生成一份。")
        return

    with st.sidebar:
        run_name = st.selectbox("选择评测记录（默认最新）", runs,
                                format_func=lambda f: f.replace(".jsonl", ""))
        run_path = os.path.join(RUNS_DIR, run_name)
        records = load_run(run_path)
        ann_path = annotations_path(run_path)
        annotated = set()
        if os.path.exists(ann_path):
            with open(ann_path, encoding="utf-8") as f:
                annotated = {json.loads(line)["case_id"] for line in f if line.strip()}
        st.metric("标注进度", f"{len(annotated)} / {len(records)}")
        st.progress(min(1.0, len(annotated) / max(1, len(records))))
        st.caption("盲评协议：请独立判断，不要参考 AI 裁判的意见（界面已隐藏）。")
        st.caption("标完一轮后运行 scripts/agreement.py 算一致率。")

    done_ids = [r["case"]["id"] for r in records if r["case"]["id"] in annotated]
    todo_ids = [r["case"]["id"] for r in records if r["case"]["id"] not in annotated]
    order = todo_ids + done_ids  # 未标注的排前面
    if not order:
        st.success("🎉 全部标注完成！运行 scripts/agreement.py 查看人机一致性。")
        return

    if "cursor" not in st.session_state or st.session_state.run_name != run_name:
        st.session_state.cursor = 0
        st.session_state.run_name = run_name
    st.session_state.cursor = min(st.session_state.cursor, len(order) - 1)
    current_id = order[st.session_state.cursor]
    record = next(r for r in records if r["case"]["id"] == current_id)
    case, target = record["case"], record["target"]

    st.subheader(f"{case['id']} · {case['category']}"
                 + ("（已标注，可修改）" if current_id in annotated else ""))
    st.markdown(f"**问题：** {case['query']}")
    if case["key_facts"]:
        st.info("参考要点：" + "；".join(case["key_facts"]))
    st.markdown(f"**应用回答：** {target['answer_text'] if target['answer_text'] else '（空/出错：' + target['error'] + '）'}")

    if target["cited_cards"]:
        with st.expander("引用的知识片段（判忠实度用，展开对照）"):
            for n, card in enumerate(target["cited_cards"], 1):
                if not card:
                    continue
                st.markdown(f"**[{n}] {card.get('title_zh', '')}**\n\n{card.get('content_zh', '')[:600]}")

    questions = REJECT_QUESTIONS if case["expect"] in ("reject", "safe") else ANSWER_QUESTIONS
    # key 绑定到具体用例：切换题目时不残留上一题的选择
    answers: dict[str, str] = {}
    for dim, label, options, hint in questions:
        answers[dim] = st.radio(label, options, index=None, key=f"{dim}-{current_id}",
                                help=hint or None,
                                horizontal=len(options) <= 3)
    notes = st.text_input("备注（可选，写你的理由或疑问）", key=f"notes-{current_id}")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ 上一条", use_container_width=True):
            st.session_state.cursor = max(0, st.session_state.cursor - 1)
            st.rerun()
    with col2:
        if st.button("⏭️ 跳到下一条未标注", use_container_width=True):
            if todo_ids:
                st.session_state.cursor = order.index(todo_ids[0])
            st.rerun()
    with col3:
        missing = [q[1][:2] for q in questions if answers.get(q[0]) is None]
        if st.button("💾 保存本条", type="primary", use_container_width=True, disabled=bool(missing)):
            save_annotation(ann_path, {
                "case_id": current_id,
                "run": run_name,
                "human": answers,
                "notes": notes,
            })
            if st.session_state.cursor < len(order) - 1:
                st.session_state.cursor += 1
            st.rerun()
        elif missing:
            st.caption("答完全部小题后可保存")


if __name__ == "__main__":
    main()
