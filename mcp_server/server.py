"""llm-eval MCP 服务器：本地 stdio 与远程 streamable-http 双传输。

- stdio（默认，本地单用户）：免钥，工具走默认租户 local；
- streamable-http（远程/多用户）：/mcp 全部要求 `Authorization: Bearer llev_…`，
  每次工具调用按请求头解析租户（评测任务与产物随租户隔离）。

运行：
  python scripts/mcp_server.py                                   # stdio
  python scripts/mcp_server.py --transport http --port 8801      # 远程（HTTPS 由反代负责）
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import Context, FastMCP

from mcp_server.tools import ToolApi, call_tool
from evalkit.service import ServiceError

mcp = FastMCP("llm-eval")
_api = ToolApi()


def _tenant_from_ctx(ctx: Context | None) -> str:
    """请求级租户解析：HTTP 传输按 Bearer Key 鉴权；stdio 本地可信 → local。"""
    request = getattr(getattr(ctx, "request_context", None), "request", None) if ctx else None
    if request is None:
        return "local"
    auth = request.headers.get("authorization", "")
    token = auth[len("Bearer "):] if auth.startswith("Bearer ") else ""
    return _api.service.authenticate(token)  # 失败抛 AUTH_REQUIRED → 稳定错误码


@mcp.tool()
def llm_eval_list_suites(ctx: Context = None) -> dict:
    """列出全部评测集（id、类型、用例数、分类目分布、内容哈希）。"""
    return call_tool(_api, "list_suites", tenant=_tenant_from_ctx(ctx))


@mcp.tool()
def llm_eval_get_suite(suite_id: str, ctx: Context = None, limit: int = 20) -> dict:
    """查看评测集详情与逐条用例（默认前 20 条）。"""
    return call_tool(_api, "get_suite", suite_id=suite_id, limit=limit)


@mcp.tool()
def llm_eval_list_targets(ctx: Context = None) -> dict:
    """列出已注册的被测应用目标（评测只能对已注册目标发起）。"""
    return call_tool(_api, "list_targets")


@mcp.tool()
def llm_eval_start_run(suite_id: str, target_id: str, version: str, ctx: Context = None,
                       limit: int | None = None, judge: bool = True,
                       budget_max_cases: int | None = None,
                       idempotency_key: str | None = None) -> dict:
    """启动一次评测（后台任务，立即返回 job_id；用 llm_eval_get_job 轮询）。"""
    return call_tool(_api, "start_run", suite_id=suite_id, target_id=target_id,
                     version=version, limit=limit, judge=judge,
                     budget_max_cases=budget_max_cases,
                     idempotency_key=idempotency_key, tenant=_tenant_from_ctx(ctx))


@mcp.tool()
def llm_eval_get_job(job_id: str, ctx: Context = None) -> dict:
    """查询任务状态与进度（queued/running/succeeded/partial/failed）。"""
    return call_tool(_api, "get_job", job_id=job_id, tenant=_tenant_from_ctx(ctx))


@mcp.tool()
def llm_eval_list_jobs(ctx: Context = None, limit: int = 20) -> dict:
    """列出本租户最近的评测任务。"""
    return call_tool(_api, "list_jobs", limit=limit, tenant=_tenant_from_ctx(ctx))


@mcp.tool()
def llm_eval_get_run(version_or_file: str, ctx: Context = None) -> dict:
    """查看一次运行的汇总（总/拒答/错误/耗时/各维度均分/分类目拆分）。"""
    return call_tool(_api, "get_run", version_or_file=version_or_file,
                     tenant=_tenant_from_ctx(ctx))


@mcp.tool()
def llm_eval_compare_runs(baseline: str, candidate: str, ctx: Context = None) -> dict:
    """版本回归对比：总体与分类目的维度差异（Δ）。"""
    return call_tool(_api, "compare_runs", baseline=baseline, candidate=candidate,
                     tenant=_tenant_from_ctx(ctx))


@mcp.tool()
def llm_eval_attribute_run(version_or_file: str, ctx: Context = None) -> dict:
    """失败两级归因（检索层 vs 生成层；仅支持含期望卡片的 RAG 套件）。"""
    return call_tool(_api, "attribute_run", version_or_file=version_or_file,
                     tenant=_tenant_from_ctx(ctx))


class BearerAuthMiddleware:
    """ASGI 鉴权包装：/mcp 路径要求 Bearer Key；lifespan 等非 HTTP 作用域原样透传。

    注意必须透传 lifespan——FastMCP 的 session manager 生命周期挂在应用 lifespan 上，
    包一层就丢会话管理（实现时踩过）。
    """

    def __init__(self, app, service):
        self.app = app
        self.service = service

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path", "").startswith("/mcp"):
            headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                       for k, v in scope.get("headers", [])}
            token = headers.get("authorization", "")
            token = token[len("Bearer "):] if token.startswith("Bearer ") else ""
            try:
                self.service.authenticate(token)
            except ServiceError as exc:
                body = f'{{"error": {{"code": "{exc.code}", "message": "{exc.message}"}}}}'.encode()
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


def create_http_app(service=None):
    """远程 MCP 的 ASGI 应用（鉴权 + streamable-http）。"""
    api = ToolApi(service=service)
    return BearerAuthMiddleware(mcp.streamable_http_app(), api.service)


def main() -> int:
    parser = argparse.ArgumentParser(description="llm-eval MCP 服务器")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8801)
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
        return 0

    import uvicorn

    uvicorn.run(create_http_app(), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
