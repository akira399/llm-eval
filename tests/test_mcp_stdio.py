"""MCP stdio 契约测试：真实拉起服务器子进程，走完整 initialize → tools/list → tools/call。

这是协议级测试：验证 MCP 客户端（如 Claude/Cursor）能发现并调用本服务。
mcp SDK 未安装时跳过。
"""
import asyncio
import os
import sys

import pytest

pytest.importorskip("mcp")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(_REPO, "scripts", "mcp_server.py")


def _run_async(coro, timeout=90):
    return asyncio.run(asyncio.wait_for(coro, timeout))


def test_stdio_handshake_tools_and_call():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = dict(os.environ)
    env["LLM_EVAL_ROOT"] = _REPO

    params = StdioServerParameters(command=sys.executable, args=[SERVER], env=env)

    async def flow():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                assert init.serverInfo.name == "llm-eval"

                tools = await session.list_tools()
                names = {t.name for t in tools.tools}
                assert {"llm_eval_list_suites", "llm_eval_start_run", "llm_eval_get_job",
                        "llm_eval_get_run", "llm_eval_compare_runs",
                        "llm_eval_attribute_run", "llm_eval_list_targets"} <= names

                result = await session.call_tool("llm_eval_list_suites", {})
                text = result.content[0].text
                assert "demo-chat" in text and "demo-json" in text

                # 未注册目标必须被拒绝（稳定错误码透传到工具结果）
                bad = await session.call_tool(
                    "llm_eval_start_run",
                    {"suite_id": "demo-chat", "target_id": "http://evil", "version": "x"})
                assert "TARGET_NOT_REGISTERED" in bad.content[0].text

    _run_async(flow())


def test_tool_api_error_wrapping():
    """工具层错误包装：业务异常 → 稳定错误码（不经协议即可验证）。"""
    from mcp_server.tools import ToolApi, call_tool

    api = ToolApi(service=None)  # 默认指向真实仓库（无副作用只读操作）
    ok = call_tool(api, "list_suites")
    assert ok["status"] == "ok" and any(s["suite_id"] == "demo-chat" for s in ok["data"]["suites"])
    bad = call_tool(api, "get_suite", suite_id="no-such")
    assert bad["status"] == "error" and bad["error"]["code"] == "SUITE_NOT_FOUND"
