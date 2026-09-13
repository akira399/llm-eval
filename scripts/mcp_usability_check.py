"""MCP 服务可用性检查（模拟真实客户端完整用户旅程）。

以 MCP 客户端身份（官方 SDK stdio 通道）拉起服务器子进程，依次验证：
  1. 协议握手 + 工具发现（9 个 llm_eval_* 工具）
  2. 只读工具：列套件 / 查详情 / 列目标
  3. 异常路径：未注册目标（TARGET_NOT_REGISTERED）、预算超限（BUDGET_EXCEEDED）
  4. 幂等：同一 idempotency_key 重复提交返回同一 job
  5. 端到端任务：demo-chat 发起 → 轮询 → 终态 → get_run
  6. 真实 RAG 目标（需 POKE_RAG_ROOT）：pokerag-local 发起 → 轮询 → get_run → attribute_run
  7. 跨进程持久化：新开第二个服务器进程（模拟客户端重连/服务重启）仍能查到任务

用法：python scripts/mcp_usability_check.py
输出：逐项 ✓/✗ 检查单；任何 ✗ 以退出码 1 结束（可作回归门禁）。
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(_REPO, "scripts", "mcp_server.py")

RESULTS: list[tuple[bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((ok, f"{'✓' if ok else '✗'} {name}" + (f" —— {detail}" if detail else "")))
    print(RESULTS[-1][1], flush=True)


def tool_result_text(result) -> str:
    return result.content[0].text if result.content else ""


def tool_data(result) -> dict:
    import json

    return json.loads(tool_result_text(result))


def make_params(env_extra: dict | None = None) -> StdioServerParameters:
    env = dict(os.environ)
    env.setdefault("LLM_EVAL_ROOT", _REPO)
    if env_extra:
        env.update(env_extra)
    return StdioServerParameters(command=sys.executable, args=[SERVER], env=env)


async def session_flow(params: StdioServerParameters, fn) -> None:
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await fn(session)


async def main() -> None:
    rag_root = os.environ.get("POKE_RAG_ROOT", "")
    key = f"usability-{os.getpid()}"

    async def journey(session):
        # 1) 握手与工具发现
        tools = await session.list_tools()
        names = {t.name for t in tools.tools}
        expected = {"llm_eval_list_suites", "llm_eval_get_suite", "llm_eval_list_targets",
                    "llm_eval_start_run", "llm_eval_get_job", "llm_eval_list_jobs",
                    "llm_eval_get_run", "llm_eval_compare_runs", "llm_eval_attribute_run"}
        check("协议握手与工具发现（9 工具）", expected <= names, f"实际 {len(names)} 个")

        # 2) 只读工具
        suites = tool_data(await session.call_tool("llm_eval_list_suites", {}))
        ids = {s["suite_id"] for s in suites["data"]["suites"]}
        check("list_suites 发现演示与 RAG 套件", {"demo-chat", "demo-json"} <= ids and
              ("poke-rag-v1" in ids), f"套件 {sorted(ids)}")
        suite = tool_data(await session.call_tool("llm_eval_get_suite",
                                                  {"suite_id": "demo-chat", "limit": 3}))
        check("get_suite 详情与分页", suite["data"]["case_count"] == 6
              and suite["data"]["cases_returned"] == 3)
        targets = tool_data(await session.call_tool("llm_eval_list_targets", {}))
        check("list_targets 列出注册目标",
              {"demo-chat", "demo-json", "pokerag-local"} <=
              {t["target_id"] for t in targets["data"]["targets"]})

        # 3) 异常路径（稳定错误码透传）
        bad_target = tool_data(await session.call_tool(
            "llm_eval_start_run",
            {"suite_id": "demo-chat", "target_id": "http://evil", "version": "x"}))
        check("未注册目标被拒绝", bad_target["error"]["code"] == "TARGET_NOT_REGISTERED")
        over = tool_data(await session.call_tool(
            "llm_eval_start_run",
            {"suite_id": "demo-chat", "target_id": "demo-chat", "version": "x",
             "budget_max_cases": 2}))
        check("预算超限硬拒绝", over["error"]["code"] == "BUDGET_EXCEEDED")

        # 4) 幂等
        a = tool_data(await session.call_tool(
            "llm_eval_start_run",
            {"suite_id": "demo-chat", "target_id": "demo-chat", "version": "usability-a",
             "idempotency_key": key + "-a"}))
        b = tool_data(await session.call_tool(
            "llm_eval_start_run",
            {"suite_id": "demo-chat", "target_id": "demo-chat", "version": "usability-a",
             "idempotency_key": key + "-a"}))
        check("幂等键重复提交返回同一任务", a["data"]["job"]["id"] == b["data"]["job"]["id"])

        # 5) 端到端任务（demo 目标，立即返回 → 轮询 → 终态 → 结果）
        started = tool_data(await session.call_tool(
            "llm_eval_start_run",
            {"suite_id": "demo-chat", "target_id": "demo-chat", "version": "usability-e2e"}))
        job_id = started["data"]["job"]["id"]
        status = "queued"
        for _ in range(60):
            job = tool_data(await session.call_tool("llm_eval_get_job", {"job_id": job_id}))["data"]
            status = job["status"]
            if status in ("succeeded", "partial", "failed", "cancelled"):
                break
            await asyncio.sleep(0.5)
        check("端到端任务到达终态", status in ("succeeded", "partial"), f"status={status}")
        run = tool_data(await session.call_tool("llm_eval_get_run",
                                                {"version_or_file": "usability-e2e"}))["data"]
        check("get_run 取回结果（6 条，维度均分齐全）",
              run["n_cases"] == 6 and 0 < run["judge_means"].get("fact_coverage", 0) <= 1.0)

        # 6) 真实 RAG 目标（可选：需要 POKE_RAG_ROOT）
        if rag_root and os.path.isdir(rag_root):
            rag = tool_data(await session.call_tool(
                "llm_eval_start_run",
                {"suite_id": "poke-rag-v0", "target_id": "pokerag-local",
                 "version": "usability-rag", "limit": 2}))
            rag_job = rag["data"]["job"]["id"]
            for _ in range(120):
                job = tool_data(await session.call_tool("llm_eval_get_job", {"job_id": rag_job}))["data"]
                status = job["status"]
                if status in ("succeeded", "partial", "failed", "cancelled"):
                    break
                await asyncio.sleep(1)
            check("真实 RAG 目标经 MCP 跑通", status in ("succeeded", "partial"), f"status={status}")
            rag_run = tool_data(await session.call_tool("llm_eval_get_run",
                                                        {"version_or_file": "usability-rag"}))["data"]
            check("RAG 运行四维评分可用", "correctness" in rag_run["judge_means"],
                  f"means={rag_run['judge_means']}")
            attr = tool_data(await session.call_tool("llm_eval_attribute_run",
                                                     {"version_or_file": "usability-rag"}))["data"]
            check("attribute_run 归因可用", "failures" in attr)
        else:
            check("真实 RAG 目标（跳过：未设 POKE_RAG_ROOT）", True)

        # 7) 对比工具
        cmp = tool_data(await session.call_tool("llm_eval_compare_runs",
                                                {"baseline": "usability-e2e", "candidate": "usability-e2e"}))
        check("compare_runs 自比 Δ=0",
              cmp["data"]["delta"]["judge_means"].get("fact_coverage", 0) == 0)

    # 第一进程：主旅程
    await session_flow(make_params(), journey)

    # 第二进程（模拟服务重启后客户端重连）：任务持久化可查
    key_row = {"job_id": ""}

    async def persistence_check(session):
        jobs = tool_data(await session.call_tool("llm_eval_list_jobs", {"limit": 50}))["data"]["jobs"]
        mine = [j for j in jobs if j["idempotency_key"] == key + "-a"]
        check("跨进程持久化（重启后任务可查）",
              bool(mine) and mine[0]["status"] in ("succeeded", "partial"),
              f"status={mine[0]['status'] if mine else 'missing'}")

    await session_flow(make_params(), persistence_check)


if __name__ == "__main__":
    asyncio.run(main())
    failed = [name for ok, name in RESULTS if not ok]
    print(f"\n===== 可用性检查：{len(RESULTS) - len(failed)}/{len(RESULTS)} 通过 =====")
    sys.exit(1 if failed else 0)
