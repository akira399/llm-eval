"""LLM-Eval 工作台（产品入口）：浏览器里点鼠标完成评测全流程。

打开方式（任选其一）：
  双击仓库根目录的 启动工作台.bat（Windows，本机已配置）
  python -m streamlit run scripts/console.py --server.port 8501

页面：总览 / 发起评测 / 任务中心 / 结果查看 / 失败归因 / 版本对比 / 人工标注。
全部走 EvalService（与 MCP/HTTP API 同一服务层），本地默认租户 local、免钥。
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from evalkit.registry import effective_catalog
from evalkit.service import EvalService, ServiceError, get_service
from evalkit.targets_store import (
    add_http_target, config_path, load_user_config, pokerag_root_override,
    remove_http_target, save_user_config)

st.set_page_config(page_title="LLM-Eval 工作台", page_icon="🧪", layout="wide")

DIM_NAMES = {"correctness": "正确性", "faithfulness": "引用忠实度", "format": "格式",
             "tone": "语气", "fact_coverage": "要点覆盖", "boundary": "边界行为",
             "json_schema_valid": "Schema 合法", "field_accuracy": "字段核对",
             "relevance": "相关性"}

PAGE = st.sidebar.radio("功能", ["🏠 总览", "🚀 发起评测", "📋 任务中心",
                                "📊 结果查看", "🔍 失败归因", "⚖️ 版本对比",
                                "⚙️ 被测目标设置", "📥 导入评测集", "✍️ 人工标注"],
                        key="page")
st.sidebar.caption("LLM-Eval · LLM 应用效果评测与回归平台")


@st.cache_resource
def _service() -> EvalService:
    # 必须经 get_service()（读 LLM_EVAL_ROOT）；直接 EvalService() 会固定到仓库根，
    # 测试/多目录部署时写错位置（实测踩坑）
    return get_service()


svc = _service()


def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except ServiceError as exc:
        return None, f"【{exc.code}】{exc}"
    except Exception as exc:
        return None, f"出错了：{type(exc).__name__}: {exc}"


def _dim_table(judge_means: dict) -> pd.DataFrame:
    return pd.DataFrame([{"维度": DIM_NAMES.get(d, d), "均分": v}
                         for d, v in judge_means.items()])


# ---------------------------------------------------------------- 总览

if PAGE == "🏠 总览":
    st.title("🧪 LLM-Eval 工作台")
    st.caption("给 AI 应用做体检：同一套考卷反复考，用数字回答「变好还是变坏、问题出在哪层、AI 裁判可不可信」。")

    suites, _ = _safe(svc.list_suites)
    jobs, _ = _safe(svc.list_jobs, limit=5)
    runs, _ = _safe(svc.list_runs, limit=5)

    c1, c2, c3 = st.columns(3)
    c1.metric("评测集（考卷）", len(suites) if suites else 0)
    c2.metric("被测目标", len(effective_catalog()))
    c3.metric("最近任务", len(jobs) if jobs else 0)

    st.subheader("📚 评测集（考卷）")
    if suites:
        st.dataframe(pd.DataFrame([{
            "评测集": s["suite_id"], "类型": s["kind"], "用例数": s["case_count"],
            "分类目": "、".join(f"{k}×{v}" for k, v in list(s["categories"].items())[:4]),
        } for s in suites]), use_container_width=True, hide_index=True)

    st.subheader("🤖 可评测的目标（被测 AI 应用）")
    st.dataframe(pd.DataFrame([{"目标": k, "说明": v["description"]}
                               for k, v in effective_catalog().items()]),
                 use_container_width=True, hide_index=True)

    st.subheader("🕐 最近的评测任务")
    if jobs:
        st.dataframe(pd.DataFrame([{
            "任务": j["id"], "状态": j["status"], "评测集": j["suite_id"],
            "目标": j["target_id"], "版本": j["version"], "进度": f"{j['done']}/{j['total']}",
            "时间": j["created_at"],
        } for j in jobs]), use_container_width=True, hide_index=True)
    else:
        st.info("还没有任务。去「🚀 发起评测」开始第一次体检吧。")

# ---------------------------------------------------------------- 发起评测

elif PAGE == "🚀 发起评测":
    st.title("🚀 发起评测")
    st.caption("选一套考卷 + 一个被测 AI 应用，点一个按钮。任务在后台跑，去「📋 任务中心」看进度。")

    suites, _ = _safe(svc.list_suites)
    suite_ids = [s["suite_id"] for s in (suites or [])]
    if not suite_ids:
        st.warning("没有可用评测集。"); st.stop()

    targets, _ = _safe(svc.list_targets)
    unavailable = {t["target_id"]: t.get("unavailable_reason", "")
                   for t in (targets or []) if not t.get("available")}
    c1, c2 = st.columns(2)
    suite_id = c1.selectbox("评测集", suite_ids, key="start-suite")
    target_options = [f"{t['target_id']}（{'可用' if t.get('available') else '不可用：配置缺失'}）"
                      for t in (targets or [])]
    target_pick = c2.selectbox("被测目标", target_options, key="start-target")
    target_id = target_pick.split("（")[0]
    if target_id in unavailable:
        st.error(f"该目标暂不可用：{unavailable[target_id]} —— 去「⚙️ 被测目标设置」补配置后再来。")
    version = st.text_input("版本标签（给这次评测起个名，之后用它对比）",
                            value=f"run-{time.strftime('%m%d-%H%M')}", key="start-version")
    meta_suite, _ = _safe(svc.get_suite, suite_id, limit=1)
    total = meta_suite["case_count"] if meta_suite else 0
    c3, c4 = st.columns(2)
    limit = c3.number_input("本轮只跑前 N 条（0 = 全部）", min_value=0, value=0, key="start-limit")
    judge = c4.checkbox("让 AI 裁判打分（关掉=只保存原始回答，零成本）", value=True, key="start-judge")
    if total:
        st.caption(f"该评测集共 {total} 条用例。")

    if st.button("🚀 开始评测", key="start-btn", type="primary", disabled=target_id in unavailable):
        with st.spinner("提交任务…"):
            job, err = _safe(svc.start_run, suite_id=suite_id, target_id=target_id,
                             version=version, limit=int(limit) or None, judge=judge,
                             budget_max_cases=max(total, 1))
        if err:
            st.error(err)
        else:
            st.success(f"任务已提交：{job['id']}（{job['total']} 条）。到「📋 任务中心」看进度。")
            st.session_state["last_job"] = job["id"]

# ---------------------------------------------------------------- 任务中心

elif PAGE == "📋 任务中心":
    st.title("📋 任务中心")
    auto = st.checkbox("自动刷新（每 2 秒，跑长任务时打开）", key="auto-refresh")
    if st.button("🔄 刷新", key="refresh-btn"):
        st.rerun()

    jobs, _ = _safe(svc.list_jobs, limit=20)
    if not jobs:
        st.info("还没有任务。"); st.stop()
    st.dataframe(pd.DataFrame([{
        "任务": j["id"], "状态": j["status"], "评测集": j["suite_id"], "目标": j["target_id"],
        "版本": j["version"], "进度": f"{j['done']}/{j['total']}", "时间": j["created_at"],
    } for j in jobs]), use_container_width=True, hide_index=True)

    job_id = st.selectbox("查看某个任务的进度", [j["id"] for j in jobs], key="job-pick")
    job, _ = _safe(svc.get_job, job_id)
    if job:
        st.progress(min(1.0, job["done"] / max(1, job["total"])),
                    text=f"{job['status']} · {job['done']}/{job['total']}")
        if job["status"] in ("succeeded", "partial"):
            st.success("跑完了！去「📊 结果查看」选这个版本看成绩；RAG 类可去「🔍 失败归因」。")
        elif job["status"] == "failed":
            st.error(f"任务失败：{job.get('error', '未知原因')}")
    if auto:
        time.sleep(2)
        st.rerun()

# ---------------------------------------------------------------- 结果查看

elif PAGE == "📊 结果查看":
    st.title("📊 结果查看")
    runs, _ = _safe(svc.list_runs)
    if not runs:
        st.info("还没有运行记录。先去「🚀 发起评测」。"); st.stop()
    run = st.selectbox("选择一次运行", [r["version"] for r in runs],
                       format_func=lambda v: next(r["mtime"] + " · " + v for r in runs if r["version"] == v),
                       key="result-run")
    summary, _ = _safe(svc.get_run, run)
    if not summary:
        st.stop()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("用例数", summary["n_cases"])
    c2.metric("正常作答", summary["n_answered"])
    c3.metric("拒答", summary["n_rejected"])
    c4.metric("错误", summary["n_errors"])
    c5.metric("平均耗时", f"{summary['latency_ms_avg'] or 0}ms")

    if summary.get("judge_means"):
        st.subheader("各维均分")
        cols = st.columns(len(summary["judge_means"]))
        for col, (dim, val) in zip(cols, summary["judge_means"].items()):
            col.metric(DIM_NAMES.get(dim, dim), round(val, 3))
        st.bar_chart(_dim_table(summary["judge_means"]).set_index("维度"))

    st.subheader("分类目成绩")
    rows = []
    for cat, s in summary["by_category"].items():
        row = {"类目": cat, "数量": s["n"], "拒答": s["n_rejected"], "错误": s["n_errors"]}
        row.update({DIM_NAMES.get(k.replace("_mean", ""), k): v
                    for k, v in s.items() if k.endswith("_mean")})
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if summary.get("llm_usage"):
        st.caption(f"本轮裁判用量：{summary['llm_usage']}")

    st.subheader("需要关注的用例")
    run_path = os.path.join(svc._runs_dir_for("local"), summary["run_file"])
    try:
        with open(run_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        bad = [r for r in records
               if r["target"].get("error")
               or (((r.get("judge") or {}).get("correctness") or {}).get("score") is not None
                   and r["judge"]["correctness"]["score"] < 0.5)]
        if not bad:
            st.success("本轮没有低分用例 🎉")
        for r in bad:
            case = r["case"]
            title = case.get("case_id") or case.get("id") or "?"
            query = case.get("query") or case.get("input", {}).get("text", "")
            with st.expander(f"{title} · {case.get('category', '')} · {query}"):
                st.markdown(f"**回答：** {r['target'].get('answer_text', '') or '（出错：' + str(r['target'].get('error')) + '）'}")
                for dim, d in (r.get("judge") or {}).items():
                    if isinstance(d, dict) and d.get("score") is not None and d["score"] < 1:
                        st.caption(f"{DIM_NAMES.get(dim, dim)}={d['score']} · "
                                   f"{d.get('reason', '') or d.get('issues', '')}")
    except FileNotFoundError:
        st.caption("原始记录文件已移动或清理。")

# ---------------------------------------------------------------- 失败归因

elif PAGE == "🔍 失败归因":
    st.title("🔍 失败归因")
    st.caption("两级定位：先看「期望的知识有没有被检索到」（检索/组装层），再看「拿到了素材有没有答对」（生成层）。目前支持 RAG 类评测集。")
    runs, _ = _safe(svc.list_runs)
    if not runs:
        st.info("还没有运行记录。"); st.stop()
    run = st.selectbox("选择一次运行", [r["version"] for r in runs], key="attr-run")
    result, err = _safe(svc.attribute_run, run)
    if err:
        st.info(err); st.stop()
    st.metric("失败用例", result["n_failures"])
    for cid, a in result["failures"].items():
        if a.get("failed"):
            with st.expander(f"{cid} · 【{a['layer']}】{a['label']}"):
                st.write(a.get("detail", ""))

# ---------------------------------------------------------------- 版本对比

elif PAGE == "⚖️ 版本对比":
    st.title("⚖️ 版本对比")
    st.caption("同一套考卷跑两个版本（改了提示词/换了模型/调了参数），看差异。")
    runs, _ = _safe(svc.list_runs)
    if len(runs) < 2:
        st.info("至少需要两次运行记录。"); st.stop()
    versions = [r["version"] for r in runs]
    c1, c2 = st.columns(2)
    base = c1.selectbox("基准版本", versions, key="cmp-base")
    cand = c2.selectbox("对比版本", [v for v in versions if v != base] or versions, index=0,
                        key="cmp-cand")
    result, err = _safe(svc.compare_runs, base, cand)
    if err:
        st.error(err); st.stop()
    left, right, delta = result["baseline"], result["candidate"], result["delta"]

    rows = []
    for dim, dv in delta["judge_means"].items():
        rows.append({"维度": DIM_NAMES.get(dim, dim),
                     f"基准({left['n_cases']}条)": left["judge_means"].get(dim, "-"),
                     f"对比({right['n_cases']}条)": right["judge_means"].get(dim, "-"),
                     "变化": dv})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("变化 > 0 表示对比版本更好；两次运行用的是同一套考卷，数字可直接比较。")
    st.write(f"拒答：基准 {left['n_rejected']} → 对比 {right['n_rejected']}；"
             f"错误：基准 {left['n_errors']} → 对比 {right['n_errors']}")

# ---------------------------------------------------------------- 被测目标设置

elif PAGE == "⚙️ 被测目标设置":
    st.title("⚙️ 被测目标设置")
    st.caption("把你自己的 AI 应用接入评测：填本地目录（RAG 类）或注册一个 HTTP 端点。配置保存在本机 targets.local.json，不入仓库。")

    st.subheader("🤖 Poke-RAG 本地目录")
    st.caption("让评测进程直接找到被测的 Poke-RAG 代码（含 src/generation/rag.py 的那个文件夹）。")
    pr_default = pokerag_root_override() or ""
    pr = st.text_input("poke-rag 根目录（绝对路径）", value=pr_default, key="cfg-pr-root")
    c1, c2 = st.columns(2)
    if c1.button("💾 保存目录", key="cfg-pr-save"):
        save_user_config(pokerag_root=pr)
        st.success("已保存，立即生效。")
    if c2.button("🔎 检查可用性", key="cfg-pr-check"):
        result, err = _safe(svc.check_target, "pokerag-local")
        if err:
            st.error(err)
        elif result["status"] == "error":
            st.error(f"调用失败：{result['error']}")
        else:
            st.success(f"可用 ✅ 示例回答：{result['output'][:80]}（{result['latency_ms']}ms）")

    st.subheader("🌐 注册 HTTP 目标（评你自己的应用）")
    st.caption("你的应用只需提供一个 HTTP 端点：接收 {input: {...}}，返回 {status, output}。协议见 README。")
    targets, _ = _safe(svc.list_targets)
    if targets:
        rows = [{"目标": t["target_id"], "类型": t["kind"], "可用": "✅" if t.get("available") else "❌",
                 "说明": t.get("description", "") + (
                     f"（{t.get('unavailable_reason', '')}）" if not t.get("available") else "")}
                for t in targets]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    http_id = c1.text_input("目标 id（小写字母/数字/-）", key="cfg-http-id", placeholder="my-app")
    http_url = c2.text_input("端点地址", key="cfg-http-url", placeholder="http://127.0.0.1:9000/invoke")
    http_desc = c3.text_input("说明（可选）", key="cfg-http-desc")
    if st.button("➕ 注册 HTTP 目标", key="cfg-http-add"):
        try:
            add_http_target(http_id, http_url, http_desc)
            st.success(f"已注册：{http_id}"); st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    user_targets = [t["target_id"] for t in load_user_config()["http_targets"]]
    if user_targets:
        st.subheader("管理自定义目标")
        for tid in user_targets:
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.write(f"🌐 {tid}")
            if c2.button("测试连接", key=f"cfg-test-{tid}"):
                result, err = _safe(svc.check_target, tid)
                if err or result["status"] == "error":
                    st.error(f"调用失败：{err or result['error']}")
                else:
                    st.success(f"可用 ✅ {result['output'][:60]}")
            if c3.button("🗑 删除", key=f"cfg-del-{tid}"):
                remove_http_target(tid)
                st.rerun()

    st.caption(f"配置文件位置：{config_path()}")

# ---------------------------------------------------------------- 导入评测集

elif PAGE == "📥 导入评测集":
    st.title("📥 导入评测集")
    st.caption("把你自己的考卷粘进来（YAML）。保存时会完整校验，格式错误会告诉你哪里不对；也可以直接把 .yaml 文件放进仓库的 suites/ 文件夹，自动识别。")

    template = """meta:
  name: my-suite
  format: generic
cases:
  - case_id: case-001
    input: {text: "你们退款政策是什么？"}
    expected_behavior:
      type: answer
      facts: ["7 天无理由退款"]
    category: 售后
  - case_id: case-002
    input: {text: "今天天气怎么样？"}
    expected_behavior:
      type: reject
    category: 边界"""
    if st.button("📋 填入模板", key="imp-template"):
        st.session_state["imp-content"] = template
        st.rerun()

    c1, c2 = st.columns([1, 2])
    suite_id = c1.text_input("评测集名（字母/数字/-/_）", key="imp-id", placeholder="my-suite")
    c2.checkbox("同名时覆盖", key="imp-overwrite")
    content = st.text_area("评测集内容（YAML）", height=380, key="imp-content",
                           value=st.session_state.get("imp-content", ""))
    if st.button("💾 校验并保存", key="imp-save", type="primary"):
        result, err = _safe(svc.save_suite, suite_id, content,
                            overwrite=st.session_state.get("imp-overwrite", False))
        if err:
            st.error(err)
        else:
            st.success(f"已导入：{result['case_count']} 条用例 → suites/{result['suite_id']}.yaml。"
                       "去「🚀 发起评测」就能选到它。")

# ---------------------------------------------------------------- 人工标注

else:
    st.title("✍️ 人工标注（盲评）")
    st.caption("给「AI 裁判可信」提供证据：你独立打分（界面不显示 AI 分数），再算人机一致率。")
    st.markdown(
        "1. 另开一个终端启动标注界面：\n\n"
        "   `python -m streamlit run scripts/annotate_app.py`\n\n"
        "2. 逐条打分保存（可分多次，进度自动记住）；\n"
        "3. 标完后运行：`python scripts/agreement.py` 查看一致率与 Kappa。")
    st.info("提示：这一步最好由人完成且不偷看 AI 分数——它是报告里「AI 裁判可信」的核心证据。")
