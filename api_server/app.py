"""llm-eval HTTP API 服务（Phase 2）：与 MCP 共用同一 ToolApi/服务层。

这是"多入口同一引擎"架构的直接证明：Web API / CLI / MCP 三种入口，
业务逻辑只有 evalkit.service 一份。响应包与 MCP 工具完全一致：
    {"status": "ok", "data": {...}} / {"status": "error", "error": {code, message}}

运行：uvicorn api_server.app:app --host 127.0.0.1 --port 8800
交互文档：http://127.0.0.1:8800/docs
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request

from mcp_server.tools import ToolApi, call_tool

HTTP_STATUS_BY_CODE = {
    "SUITE_NOT_FOUND": 404, "RUN_NOT_FOUND": 404, "JOB_NOT_FOUND": 404,
    "ARTIFACT_NOT_FOUND": 404,
    "TARGET_NOT_REGISTERED": 400, "BUDGET_EXCEEDED": 400, "UNSUPPORTED": 400,
    "INVALID_ARGUMENT": 400,
}


def create_app(service=None) -> FastAPI:
    """app 工厂：生产用模块级 app（env 指定 root）；测试注入独立 service。"""
    app = FastAPI(title="llm-eval API", version="0.1.0",
                  description="LLM 应用效果评测与回归平台（与 MCP 共用同一服务层）")
    api = ToolApi(service=service)

    def respond(fn_name: str, ok_status: int = 200, **kwargs) -> dict:
        envelope = call_tool(api, fn_name, **kwargs)
        if envelope["status"] != "ok":
            code = envelope["error"]["code"]
            raise HTTPException(status_code=HTTP_STATUS_BY_CODE.get(code, 500), detail=envelope["error"])
        return envelope["data"]

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "service": "llm-eval"}

    @app.get("/v1/suites")
    def list_suites() -> dict:
        return respond("list_suites")

    @app.get("/v1/suites/{suite_id}")
    def get_suite(suite_id: str, limit: int = 20) -> dict:
        return respond("get_suite", suite_id=suite_id, limit=limit)

    @app.get("/v1/targets")
    def list_targets() -> dict:
        return respond("list_targets")

    @app.post("/v1/jobs")
    def start_job(body: dict, request: Request) -> dict:
        for field in ("suite_id", "target_id", "version"):
            if not body.get(field):
                raise HTTPException(status_code=400, detail={
                    "code": "INVALID_ARGUMENT", "message": f"缺少必填字段 {field}"})
        idem = request.headers.get("Idempotency-Key")
        return respond("start_run", suite_id=body["suite_id"], target_id=body["target_id"],
                       version=body["version"], limit=body.get("limit"),
                       judge=bool(body.get("judge", True)),
                       budget_max_cases=body.get("budget_max_cases"),
                       idempotency_key=idem)

    @app.get("/v1/jobs")
    def list_jobs(limit: int = 20) -> dict:
        return respond("list_jobs", limit=limit)

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        return respond("get_job", job_id=job_id)

    @app.post("/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict:
        return respond("cancel_job", job_id=job_id)

    @app.get("/v1/runs/{version_or_file}")
    def get_run(version_or_file: str) -> dict:
        return respond("get_run", version_or_file=version_or_file)

    @app.get("/v1/compare")
    def compare(baseline: str, candidate: str) -> dict:
        return respond("compare_runs", baseline=baseline, candidate=candidate)

    @app.get("/v1/attribute/{version_or_file}")
    def attribute(version_or_file: str) -> dict:
        return respond("attribute_run", version_or_file=version_or_file)

    return app


app = create_app()
