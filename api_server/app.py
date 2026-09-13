"""llm-eval HTTP API 服务：与 MCP 共用同一 ToolApi/服务层。

访问控制（Phase 2.5）：
- auth_enabled=True（生产默认）时，/v1/* 全部要求 `Authorization: Bearer llev_…`；
  Key → 租户由 EvalService.authenticate 解析，租户隔离由服务层保证；
- 本地/测试可 auth_enabled=False（等同于默认租户 local）；
- /api/health 永远免鉴权（探活）。

运行：uvicorn api_server.app:app --host 127.0.0.1 --port 8800
交互文档：http://127.0.0.1:8800/docs
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request

from mcp_server.tools import ToolApi, call_tool
from evalkit.service import EvalService, ServiceError

HTTP_STATUS_BY_CODE = {
    "SUITE_NOT_FOUND": 404, "RUN_NOT_FOUND": 404, "JOB_NOT_FOUND": 404,
    "ARTIFACT_NOT_FOUND": 404,
    "TARGET_NOT_REGISTERED": 400, "BUDGET_EXCEEDED": 400, "UNSUPPORTED": 400,
    "INVALID_ARGUMENT": 400, "AUTH_REQUIRED": 401,
}


def create_app(service: EvalService | None = None, auth_enabled: bool = True) -> FastAPI:
    app = FastAPI(title="llm-eval API", version="0.2.0",
                  description="LLM 应用效果评测与回归平台（MCP/API/CLI 共用同一服务层）")
    api = ToolApi(service=service)

    def tenant_of(request: Request) -> str:
        if not auth_enabled:
            return "local"
        auth = request.headers.get("authorization", "")
        token = auth[len("Bearer "):] if auth.startswith("Bearer ") else ""
        try:
            return service.authenticate(token)
        except ServiceError as exc:
            raise HTTPException(status_code=401, detail={"code": exc.code, "message": str(exc)})

    def respond(fn_name: str, **kwargs) -> dict:
        envelope = call_tool(api, fn_name, **kwargs)
        if envelope["status"] != "ok":
            code = envelope["error"]["code"]
            raise HTTPException(status_code=HTTP_STATUS_BY_CODE.get(code, 500), detail=envelope["error"])
        return envelope["data"]

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "service": "llm-eval"}

    @app.get("/v1/suites")
    def list_suites(request: Request) -> dict:
        tenant_of(request)  # 统一鉴权：除健康检查外所有 /v1/* 都要求有效 Key
        return respond("list_suites")

    @app.get("/v1/suites/{suite_id}")
    def get_suite(suite_id: str, request: Request, limit: int = 20) -> dict:
        tenant_of(request)
        return respond("get_suite", suite_id=suite_id, limit=limit)

    @app.get("/v1/targets")
    def list_targets(request: Request) -> dict:
        tenant_of(request)
        return respond("list_targets")

    @app.post("/v1/jobs")
    def start_job(body: dict, request: Request) -> dict:
        tenant = tenant_of(request)
        for field in ("suite_id", "target_id", "version"):
            if not body.get(field):
                raise HTTPException(status_code=400, detail={
                    "code": "INVALID_ARGUMENT", "message": f"缺少必填字段 {field}"})
        idem = request.headers.get("Idempotency-Key")
        return respond("start_run", suite_id=body["suite_id"], target_id=body["target_id"],
                       version=body["version"], limit=body.get("limit"),
                       judge=bool(body.get("judge", True)),
                       budget_max_cases=body.get("budget_max_cases"),
                       idempotency_key=idem, tenant=tenant)

    @app.get("/v1/jobs")
    def list_jobs(request: Request, limit: int = 20) -> dict:
        return respond("list_jobs", limit=limit, tenant=tenant_of(request))

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str, request: Request) -> dict:
        return respond("get_job", job_id=job_id, tenant=tenant_of(request))

    @app.post("/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, request: Request) -> dict:
        return respond("cancel_job", job_id=job_id, tenant=tenant_of(request))

    @app.get("/v1/runs/{version_or_file}")
    def get_run(version_or_file: str, request: Request) -> dict:
        return respond("get_run", version_or_file=version_or_file, tenant=tenant_of(request))

    @app.get("/v1/compare")
    def compare(baseline: str, candidate: str, request: Request) -> dict:
        return respond("compare_runs", baseline=baseline, candidate=candidate,
                       tenant=tenant_of(request))

    @app.get("/v1/attribute/{version_or_file}")
    def attribute(version_or_file: str, request: Request) -> dict:
        return respond("attribute_run", version_or_file=version_or_file, tenant=tenant_of(request))

    return app


app = create_app()
