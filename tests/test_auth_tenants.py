"""租户隔离与 API Key 鉴权测试（Phase 2.5）。"""
import os
import shutil

import pytest

from fastapi.testclient import TestClient

from api_server.app import create_app
from evalkit.service import EvalService, ServiceError

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def service(tmp_path):
    os.makedirs(os.path.join(str(tmp_path), "suites"), exist_ok=True)
    shutil.copy(os.path.join(_REPO, "suites", "demo-chat.yaml"),
                os.path.join(str(tmp_path), "suites", "demo-chat.yaml"))
    return EvalService(root=str(tmp_path))


def _run_job(service, tenant, version):
    job = service.start_run(suite_id="demo-chat", target_id="demo-chat",
                            version=version, tenant=tenant, async_start=False)
    service._execute_job(job["id"])
    return service.get_job(job["id"], tenant=tenant)


def test_create_tenant_and_authenticate(service):
    created = service.create_tenant("团队A")
    assert created["api_key"].startswith("llev_")
    # 明文只返回一次：库里应为哈希
    import sqlite3

    with sqlite3.connect(service.db_path) as conn:
        rows = conn.execute("SELECT key_hash FROM api_keys").fetchall()
    assert all(created["api_key"] not in r[0] for r in rows)
    assert service.authenticate(created["api_key"]) == created["tenant_id"]
    with pytest.raises(ServiceError) as ei:
        service.authenticate("llev_wrong")
    assert ei.value.code == "AUTH_REQUIRED"
    with pytest.raises(ServiceError):
        service.authenticate(None)
    # 吊销后立即失效
    service.revoke_api_key(created["api_key"])
    with pytest.raises(ServiceError):
        service.authenticate(created["api_key"])


def test_tenant_job_isolation(service):
    a = service.create_tenant("A")
    b = service.create_tenant("B")
    job = _run_job(service, a["tenant_id"], "tenant-a-run")

    assert service.get_job(job["id"], tenant=a["tenant_id"])["status"] == "succeeded"
    with pytest.raises(ServiceError) as ei:  # B 看不到 A 的任务（不泄露存在性）
        service.get_job(job["id"], tenant=b["tenant_id"])
    assert ei.value.code == "JOB_NOT_FOUND"
    assert job["id"] not in {j["id"] for j in service.list_jobs(tenant=b["tenant_id"])}


def test_tenant_run_isolation(service):
    a = service.create_tenant("A")
    b = service.create_tenant("B")
    _run_job(service, a["tenant_id"], "iso-run")

    assert service.get_run("iso-run", tenant=a["tenant_id"])["n_cases"] == 6
    with pytest.raises(ServiceError) as ei:
        service.get_run("iso-run", tenant=b["tenant_id"])
    assert ei.value.code == "RUN_NOT_FOUND"
    # 物理隔离：产物在租户子目录
    assert os.path.isdir(os.path.join(service.runs_dir, "tenants", a["tenant_id"]))


def test_idempotency_key_scoped_to_tenant(service):
    """同 key 不同租户互不冲突（幂等键按租户作用域）。"""
    a = service.create_tenant("A")
    b = service.create_tenant("B")
    ja = service.start_run(suite_id="demo-chat", target_id="demo-chat", version="v",
                           idempotency_key="k1", tenant=a["tenant_id"], async_start=False)
    jb = service.start_run(suite_id="demo-chat", target_id="demo-chat", version="v",
                           idempotency_key="k1", tenant=b["tenant_id"], async_start=False)
    assert ja["id"] != jb["id"]


def test_api_requires_bearer_key(service):
    client = TestClient(create_app(service=service, auth_enabled=True))
    assert client.get("/v1/suites").status_code == 401  # 无 Key
    assert client.get("/v1/suites",
                      headers={"Authorization": "Bearer llev_bad"}).status_code == 401
    assert client.get("/api/health").status_code == 200  # 探活免鉴权


def test_api_tenant_scoped_access(service):
    a = service.create_tenant("A")
    b = service.create_tenant("B")
    client = TestClient(create_app(service=service, auth_enabled=True))
    auth_a = {"Authorization": f"Bearer {a['api_key']}"}
    auth_b = {"Authorization": f"Bearer {b['api_key']}"}

    started = client.post("/v1/jobs", json={
        "suite_id": "demo-chat", "target_id": "demo-chat", "version": "api-t"},
        headers=auth_a)
    assert started.status_code == 200
    job_id = started.json()["job"]["id"]
    service._execute_job(job_id)

    assert client.get(f"/v1/jobs/{job_id}", headers=auth_a).status_code == 200
    assert client.get(f"/v1/jobs/{job_id}", headers=auth_b).status_code == 404
    assert client.get("/v1/runs/api-t", headers=auth_a).status_code == 200
    assert client.get("/v1/runs/api-t", headers=auth_b).status_code == 404


def test_mcp_local_mode_unaffected(service, tmp_path, monkeypatch):
    """本地 MCP（默认 local 租户、无 Key）照常工作。"""
    from mcp_server.tools import ToolApi, call_tool

    api = ToolApi(service=service)
    ok = call_tool(api, "start_run", suite_id="demo-chat", target_id="demo-chat",
                   version="mcp-local", async_start=False)
    assert ok["status"] == "ok"
    call_tool(api, "get_job", job_id=ok["data"]["job"]["id"])
    service._execute_job(ok["data"]["job"]["id"])
    run = call_tool(api, "get_run", version_or_file="mcp-local")
    assert run["status"] == "ok" and run["data"]["n_cases"] == 6
