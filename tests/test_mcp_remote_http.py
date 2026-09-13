"""远程 Streamable HTTP MCP 契约测试（Phase 4-lite）。

真实子进程起 HTTP 服务器，用官方 streamablehttp_client 验证：
无钥 401 / 有钥握手与工具调用 / 任务与产物随租户隔离。
"""
import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import time

import pytest

pytest.importorskip("mcp")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(_REPO, "scripts", "mcp_server.py")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_port(port: int, timeout: float = 30) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"MCP HTTP 服务器未在 {timeout}s 内监听端口 {port}")


@pytest.fixture(scope="module")
def http_env(tmp_path_factory):
    """独立 root + 两个租户 + HTTP 服务器子进程。"""
    root = tmp_path_factory.mktemp("mcp-http-root")
    os.makedirs(os.path.join(str(root), "suites"), exist_ok=True)
    shutil.copy(os.path.join(_REPO, "suites", "demo-chat.yaml"),
                os.path.join(str(root), "suites", "demo-chat.yaml"))
    from evalkit.service import EvalService

    service = EvalService(root=str(root))
    tenant_a = service.create_tenant("A")
    tenant_b = service.create_tenant("B")

    port = _free_port()
    env = dict(os.environ)
    env["LLM_EVAL_ROOT"] = str(root)
    proc = subprocess.Popen(
        [sys.executable, SERVER, "--transport", "http", "--port", str(port)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _wait_port(port)
    yield {"url": f"http://127.0.0.1:{port}/mcp", "service": service,
           "key_a": tenant_a["api_key"], "key_b": tenant_b["api_key"]}
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _run_async(coro, timeout=90):
    return asyncio.run(asyncio.wait_for(coro, timeout))


def test_remote_mcp_full_journey(http_env):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async def flow():
        headers = {"Authorization": f"Bearer {http_env['key_a']}"}
        async with streamablehttp_client(http_env["url"], headers=headers) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert {t.name for t in tools.tools} >= {"llm_eval_list_suites",
                                                         "llm_eval_start_run", "llm_eval_get_job"}

                suites = json.loads((await session.call_tool(
                    "llm_eval_list_suites", {})).content[0].text)
                assert any(s["suite_id"] == "demo-chat" for s in suites["data"]["suites"])

                started = json.loads((await session.call_tool("llm_eval_start_run", {
                    "suite_id": "demo-chat", "target_id": "demo-chat",
                    "version": "remote-e2e"})).content[0].text)
                job_id = started["data"]["job"]["id"]
                for _ in range(60):
                    job = json.loads((await session.call_tool(
                        "llm_eval_get_job", {"job_id": job_id})).content[0].text)["data"]
                    if job["status"] in ("succeeded", "partial", "failed"):
                        break
                    await asyncio.sleep(0.5)
                assert job["status"] == "succeeded"
                run = json.loads((await session.call_tool(
                    "llm_eval_get_run", {"version_or_file": "remote-e2e"})).content[0].text)["data"]
                assert run["n_cases"] == 6
                return job_id

    job_id = _run_async(flow())
    # 租户隔离落在服务层：B 租户查 A 的任务/结果 → 稳定错误码（不泄露存在性）
    codes = _run_async(_cross_tenant_check(http_env, job_id))
    assert codes == ("JOB_NOT_FOUND", "RUN_NOT_FOUND")


async def _cross_tenant_check(http_env, job_id):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    headers = {"Authorization": f"Bearer {http_env['key_b']}"}
    async with streamablehttp_client(http_env["url"], headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            job = json.loads((await session.call_tool(
                "llm_eval_get_job", {"job_id": job_id})).content[0].text)
            run = json.loads((await session.call_tool(
                "llm_eval_get_run", {"version_or_file": "remote-e2e"})).content[0].text)
            return job["error"]["code"], run["error"]["code"]


def test_remote_mcp_requires_key(http_env):
    """无钥/错钥：MCP 客户端连接即被 401 拒绝（握手失败可见）。"""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async def flow(headers):
        async with streamablehttp_client(http_env["url"], headers=headers) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()  # 应在此前被 401 拒绝

    with pytest.raises(Exception):
        _run_async(flow({}))
    with pytest.raises(Exception):
        _run_async(flow({"Authorization": "Bearer llev_wrong"}))
