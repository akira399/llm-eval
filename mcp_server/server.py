"""本地 stdio MCP 服务器（Phase 1）：让 MCP 客户端（Claude/Cursor 等）驱动评测平台。

运行：python scripts/mcp_server.py          （stdio 传输，客户端以子进程方式拉起）
配置示例（MCP 客户端 JSON）：
  {"command": "<python>", "args": ["<仓库>/scripts/mcp_server.py"],
   "env": {"LLM_EVAL_ROOT": "<llm-eval 仓库路径>"}}
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mcp_server.tools import ToolApi, call_tool

mcp = FastMCP("llm-eval")
_api = ToolApi()


@mcp.tool()
def llm_eval_list_suites() -> dict:
    """列出全部评测集（id、类型、用例数、分类目分布、内容哈希）。"""
    return call_tool(_api, "list_suites")


@mcp.tool()
def llm_eval_get_suite(suite_id: str, limit: int = 20) -> dict:
    """查看评测集详情与逐条用例（默认前 20 条）。"""
    return call_tool(_api, "get_suite", suite_id=suite_id, limit=limit)


@mcp.tool()
def llm_eval_list_targets() -> dict:
    """列出已注册的被测应用目标（评测只能对已注册目标发起）。"""
    return call_tool(_api, "list_targets")


@mcp.tool()
def llm_eval_start_run(suite_id: str, target_id: str, version: str,
                       limit: int | None = None, judge: bool = True,
                       budget_max_cases: int | None = None,
                       idempotency_key: str | None = None) -> dict:
    """启动一次评测（后台任务，立即返回 job_id；用 llm_eval_get_job 轮询）。"""
    return call_tool(_api, "start_run", suite_id=suite_id, target_id=target_id,
                     version=version, limit=limit, judge=judge,
                     budget_max_cases=budget_max_cases, idempotency_key=idempotency_key)


@mcp.tool()
def llm_eval_get_job(job_id: str) -> dict:
    """查询任务状态与进度（queued/running/succeeded/partial/failed）。"""
    return call_tool(_api, "get_job", job_id=job_id)


@mcp.tool()
def llm_eval_list_jobs(limit: int = 20) -> dict:
    """列出最近的评测任务。"""
    return call_tool(_api, "list_jobs", limit=limit)


@mcp.tool()
def llm_eval_get_run(version_or_file: str) -> dict:
    """查看一次运行的汇总（总/拒答/错误/耗时/各维度均分/分类目拆分）。"""
    return call_tool(_api, "get_run", version_or_file=version_or_file)


@mcp.tool()
def llm_eval_compare_runs(baseline: str, candidate: str) -> dict:
    """版本回归对比：总体与分类目的维度差异（Δ）。"""
    return call_tool(_api, "compare_runs", baseline=baseline, candidate=candidate)


@mcp.tool()
def llm_eval_attribute_run(version_or_file: str) -> dict:
    """失败两级归因（检索层 vs 生成层；仅支持含期望卡片的 RAG 套件）。"""
    return call_tool(_api, "attribute_run", version_or_file=version_or_file)


if __name__ == "__main__":
    mcp.run(transport="stdio")
