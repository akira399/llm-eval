"""Phase 2 测试：HTTP JSON 连接器 + FastAPI 服务（与 MCP 共用同一服务层的证明）。"""
import os
import shutil
import threading
import time

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from api_server.app import create_app  # noqa: E402
from evalkit.contracts import InvocationRequest, load_evaluation_cases  # noqa: E402
from evalkit.engine import EvaluationEngine  # noqa: E402
from evalkit.judge_profile import chat_profile  # noqa: E402
from evalkit.providers.http_json import HttpJsonAdapter  # noqa: E402
from evalkit.service import EvalService  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def demo_http_url():
    """真实回环 HTTP 服务（临时端口）：比 mock 传输更贴近部署形态。"""
    import uvicorn

    from scripts.demo_chat_http import app as demo_app

    config = uvicorn.Config(demo_app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}/invoke"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def http_adapter(demo_http_url) -> HttpJsonAdapter:
    return HttpJsonAdapter(base_url=demo_http_url, target_id="demo-chat-http")


@pytest.fixture
def service(tmp_path):
    os.makedirs(os.path.join(str(tmp_path), "suites"), exist_ok=True)
    shutil.copy(os.path.join(_REPO, "suites", "demo-chat.yaml"),
                os.path.join(str(tmp_path), "suites", "demo-chat.yaml"))
    return EvalService(root=str(tmp_path))


def test_connector_rejects_bad_base_url():
    with pytest.raises(ValueError):
        HttpJsonAdapter(base_url="not-a-url")
    with pytest.raises(ValueError):
        HttpJsonAdapter(base_url="http://169.254.169.254/latest")  # 云元数据地址被拒


def test_http_connector_end_to_end(http_adapter):
    obs = http_adapter.invoke(InvocationRequest(input={"text": "退款政策是什么"}))
    assert obs.status == "success" and "7 天无理由退款" in obs.output
    refused = http_adapter.invoke(InvocationRequest(input={"text": "今天天气如何"}))
    assert refused.status == "success" and "抱歉" in refused.output  # 礼貌拒答透传


def test_http_connector_error_isolation(demo_http_url):
    """404/断网 → 单条 error 观察，不抛异常、不拖垮整批。"""
    broken = HttpJsonAdapter(base_url=demo_http_url.replace("/invoke", "/nope"))
    err = broken.invoke(InvocationRequest(input={"text": "退款"}))
    assert err.status == "error" and err.output == ""

    dead = HttpJsonAdapter(base_url="http://127.0.0.1:1/invoke")  # 无监听端口
    err = dead.invoke(InvocationRequest(input={"text": "退款"}))
    assert err.status == "error"


def test_http_target_through_engine(tmp_path, http_adapter):
    """HTTP 目标 → 通用引擎 全链路（与进程内目标同一套跑法）。"""
    _, cases = load_evaluation_cases(os.path.join(_REPO, "suites", "demo-chat.yaml"))
    engine = EvaluationEngine(adapter=http_adapter, profile=chat_profile(), out_dir=str(tmp_path))
    summary = engine.run(cases, version="http-e2e")
    assert summary["n_cases"] == 6 and summary["n_errors"] == 0
    assert summary["judge_means"]["fact_coverage"] == 1.0


def test_api_service_end_to_end(service):
    """API 服务完整旅程：建任务 → 轮询 → 查结果 → 对比（与 MCP 相同的服务层）。"""
    client = TestClient(create_app(service=service, auth_enabled=False))

    assert client.get("/api/health").json()["status"] == "ok"
    suites = client.get("/v1/suites").json()["suites"]
    assert {"demo-chat"} <= {s["suite_id"] for s in suites}

    started = client.post("/v1/jobs", json={
        "suite_id": "demo-chat", "target_id": "demo-chat",
        "version": "api-e2e", "budget_max_cases": 10,
    }, headers={"Idempotency-Key": "api-001"})
    assert started.status_code == 200
    job_id = started.json()["job"]["id"]

    for _ in range(120):  # 后台线程已在跑：轮询到终态（避免同步执行竞态）
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "partial", "failed", "cancelled"):
            break
        time.sleep(0.25)
    assert job["status"] == "succeeded" and job["done"] == 6

    run = client.get("/v1/runs/api-e2e").json()
    assert run["n_cases"] == 6 and run["judge_means"]["fact_coverage"] == 1.0

    cmp = client.get("/v1/compare", params={"baseline": "api-e2e", "candidate": "api-e2e"}).json()
    assert cmp["delta"]["judge_means"]["fact_coverage"] == 0.0


def test_api_error_codes_over_http(service):
    client = TestClient(create_app(service=service, auth_enabled=False))
    assert client.get("/v1/suites/nope").status_code == 404
    bad = client.post("/v1/jobs", json={
        "suite_id": "demo-chat", "target_id": "http://evil", "version": "x"})
    assert bad.status_code == 400
    assert bad.json()["detail"]["code"] == "TARGET_NOT_REGISTERED"
    missing = client.post("/v1/jobs", json={"suite_id": "demo-chat"})
    assert missing.status_code == 400
