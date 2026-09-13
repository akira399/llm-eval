"""MCP 工具层：薄封装 EvalService，全部返回 JSON 可序列化的 dict。

设计纪律（docs/00 §11）：
- 工具不写业务逻辑，异常统一转 {error: {code, message}}（稳定错误码）；
- 只接受 suite_id/target_id/run_id 等标识符，绝不接受文件路径或任意 URL；
- 长任务工具（start_run）立即返回 job，绝不阻塞等待全量评测；
- 输出内容默认摘要级（get_run 不逐条返回原文，避免撑爆客户端上下文）。
"""
from __future__ import annotations

from evalkit.service import EvalService, ServiceError, get_service


class ToolApi:
    def __init__(self, service: EvalService | None = None):
        self.service = service or get_service()

    # ---- 只读 ----

    def list_suites(self, tenant: str = "local") -> dict:
        # 套件为项目级共享资产（全局可读）；tenant 仅为调用签名统一
        return {"suites": self.service.list_suites()}

    def get_suite(self, suite_id: str, limit: int = 20) -> dict:
        return self.service.get_suite(suite_id, limit=limit)

    def list_targets(self) -> dict:
        return {"targets": [{"target_id": tid, **info} for tid, info in TARGET_CATALOG_ITEMS()]}

    def get_run(self, version_or_file: str, tenant: str = "local") -> dict:
        return self.service.get_run(version_or_file, tenant=tenant)

    def compare_runs(self, baseline: str, candidate: str, tenant: str = "local") -> dict:
        return self.service.compare_runs(baseline, candidate, tenant=tenant)

    def attribute_run(self, version_or_file: str, tenant: str = "local") -> dict:
        return self.service.attribute_run(version_or_file, tenant=tenant)

    def list_jobs(self, limit: int = 20, tenant: str = "local") -> dict:
        return {"jobs": self.service.list_jobs(limit=limit, tenant=tenant)}

    def get_job(self, job_id: str, tenant: str = "local") -> dict:
        return self.service.get_job(job_id, tenant=tenant)

    # ---- 长任务 ----

    def start_run(self, suite_id: str, target_id: str, version: str, limit: int | None = None,
                  judge: bool = True, budget_max_cases: int | None = None,
                  idempotency_key: str | None = None, tenant: str = "local",
                  async_start: bool = True) -> dict:
        job = self.service.start_run(
            suite_id=suite_id, target_id=target_id, version=version, limit=limit,
            judge=judge, budget_max_cases=budget_max_cases, idempotency_key=idempotency_key,
            tenant=tenant, async_start=async_start,
        )
        return {"job": job, "hint": "用 get_job 轮询进度；终态后用 get_run/compare_runs/attribute_run 查看结果"}

    def cancel_job(self, job_id: str, tenant: str = "local") -> dict:
        return {"job": self.service.cancel_job(job_id, tenant=tenant)}


def TARGET_CATALOG_ITEMS():
    from evalkit.registry import TARGET_CATALOG

    return TARGET_CATALOG.items()


def call_tool(api: ToolApi, fn_name: str, **kwargs) -> dict:
    """统一错误包装：业务异常 → 稳定错误码；供 server 与测试共用。"""
    try:
        return {"status": "ok", "data": getattr(api, fn_name)(**kwargs)}
    except ServiceError as exc:
        return {"status": "error", "error": {"code": exc.code, "message": str(exc)}}
    except Exception as exc:  # 未预期异常：不泄露堆栈，只给类型与信息
        return {"status": "error", "error": {"code": "INTERNAL_ERROR",
                                             "message": f"{type(exc).__name__}: {exc}"}}
