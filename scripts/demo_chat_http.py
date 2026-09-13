"""演示被测应用（HTTP 版）：把 DemoChatAdapter 暴露成 HTTP JSON 端点。

用于验证通用 HTTP 连接器（Phase 2）：被测应用与评测平台跨进程部署的标准形态。
运行：python scripts/demo_chat_http.py            （默认 127.0.0.1:8766）
协议：POST /invoke {"input": {"text": "..."}} → {"status","output",...}
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel

from evalkit.demo import DemoChatAdapter

app = FastAPI(title="demo-chat-http", description="llm-eval 演示被测应用（HTTP JSON 协议）")
_adapter = DemoChatAdapter()


class InvokeBody(BaseModel):
    input: dict = {}
    variables: dict = {}
    metadata: dict = {}


@app.post("/invoke")
def invoke(body: InvokeBody) -> dict:
    from evalkit.contracts import InvocationRequest

    t0 = time.perf_counter()
    obs = _adapter.invoke(InvocationRequest(
        input=body.input, variables=body.variables, metadata=body.metadata))
    return {
        "status": obs.status,
        "output": obs.output,
        "structured_output": obs.structured_output,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "target": "demo-chat-http"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("DEMO_CHAT_HTTP_PORT", "8766")))
