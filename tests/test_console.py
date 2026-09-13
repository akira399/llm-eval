"""工作台 UI 测试：用官方 AppTest 驱动真实页面，验证产品核心旅程。

覆盖：总览加载 / 从「发起评测」页点按钮建任务 / 任务终态后「结果查看」
「版本对比」「失败归因」页可用（demo 目标，全离线确定性）。
"""
import os
import time

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from evalkit.service import EvalService  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONSOLE = os.path.join(_REPO, "scripts", "console.py")


@pytest.fixture
def service(tmp_path):
    os.makedirs(os.path.join(str(tmp_path), "suites"), exist_ok=True)
    for name in ("demo-chat.yaml", "demo-json.yaml"):
        shutil_copy = __import__("shutil").copy
        shutil_copy(os.path.join(_REPO, "suites", name),
                    os.path.join(str(tmp_path), "suites", name))
    return EvalService(root=str(tmp_path))


def _app(service) -> AppTest:
    """AppTest 注入临时 root 的服务（console 用 get_service()，经环境变量控制）。"""
    import streamlit as st

    st.cache_resource.clear()  # 防止进程内其他测试缓存的 EvalService 串根目录
    os.environ["LLM_EVAL_ROOT"] = str(_last_root[0])
    at = AppTest.from_file(CONSOLE, default_timeout=30)
    return at


_last_root = [os.environ.get("LLM_EVAL_ROOT", _REPO)]


@pytest.fixture(autouse=True)
def _bind_root(service, monkeypatch):
    """让 console 的 get_service() 拿到测试 root（每次 AppTest.run 前生效）。"""
    monkeypatch.setenv("LLM_EVAL_ROOT", str(service.root))
    _last_root[0] = str(service.root)


def test_overview_page_loads(service):
    at = _app(service)
    at.run()
    assert not at.exception
    texts = ([b.body for b in at.markdown] + [b.body for b in at.subheader]
             + [m.label for m in at.metric])
    assert any("评测集（考卷）" in t for t in texts)
    from evalkit.registry import TARGET_CATALOG

    assert any(m.label == "被测目标" and m.value == str(len(TARGET_CATALOG))
               for m in at.metric)


def test_start_run_from_ui_and_finish(service):
    at = _app(service)
    at.run()
    assert not at.exception

    at.sidebar.radio[0].set_value("🚀 发起评测").run()
    at.selectbox(key="start-suite").set_value("demo-chat").run()
    at.selectbox(key="start-target").set_value("demo-chat").run()
    at.text_input(key="start-version").set_value("ui-e2e").run()
    at.button(key="start-btn").click().run()
    assert not at.exception
    assert any("任务已提交" in s.value for s in at.success)

    # 后台线程跑 demo 套件（秒级）；轮询到终态
    job_id = service.list_jobs()[0]["id"]
    for _ in range(60):
        if service.get_job(job_id)["status"] in ("succeeded", "partial", "failed"):
            break
        time.sleep(0.5)
    assert service.get_job(job_id)["status"] == "succeeded"

    # 结果页能看到这次运行
    at.sidebar.radio[0].set_value("📊 结果查看").run()
    at.selectbox(key="result-run").set_value("ui-e2e").run()
    assert not at.exception
    assert any(m.label == "用例数" and m.value == 6 for m in at.metric) or            any(m.label == "用例数" and m.value == "6" for m in at.metric)
    assert any(m.label == "要点覆盖" for m in at.metric)


def test_compare_page_after_two_runs(service):
    for version in ("ui-cmp-a", "ui-cmp-b"):
        job = service.start_run(suite_id="demo-chat", target_id="demo-chat",
                                version=version, async_start=False)
        service._execute_job(job["id"])

    at = _app(service)
    at.run()
    at.sidebar.radio[0].set_value("⚖️ 版本对比").run()
    at.selectbox(key="cmp-base").set_value("ui-cmp-a").run()
    at.selectbox(key="cmp-cand").set_value("ui-cmp-b").run()
    assert not at.exception
    assert any("变化" in df.value.columns for df in at.dataframe)


def test_attribution_page_friendly_for_generic(service):
    job = service.start_run(suite_id="demo-chat", target_id="demo-chat",
                            version="ui-attr", async_start=False)
    service._execute_job(job["id"])
    at = _app(service)
    at.run()
    at.sidebar.radio[0].set_value("🔍 失败归因").run()
    at.selectbox(key="attr-run").set_value("ui-attr").run()
    assert not at.exception  # generic 套件 → 友好提示（info），不报错
    assert any("RAG" in i.value for i in at.info)
