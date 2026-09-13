"""通用 HTTP JSON 连接器（Phase 2）：被测应用只需暴露一个 HTTP 端点即可接入评测。

协议（被测应用侧实现）：
    POST <base_url>
    请求：{"input": {...}, "metadata": {...}, "variables": {...}}
    响应：{"status": "success|rejected|error", "output": "...",
           "structured_output": {...}?, "latency_ms": 123}?

安全边界（docs/00 §11）：
- base_url 只来自服务端注册配置（环境变量/registry），绝不接受客户端传参——
  这是 SSRF 的第一道防线；本连接器再校验 scheme，并拒绝云元数据地址。
- 连接失败/非 200/非 JSON 一律降级为该条用例的 error 观察（隔离，不拖垮整批）。
"""
from __future__ import annotations

from evalkit.contracts import InvocationRequest, TargetObservation, TargetAdapter

_BLOCKED_HOSTS = ("169.254.169.254", "metadata.google.internal")  # 云元数据


class HttpJsonAdapter(TargetAdapter):
    def __init__(self, base_url: str, target_id: str = "http-json", timeout_s: int = 120,
                 transport=None):
        self.target_id = target_id
        self.base_url = base_url.rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError(f"base_url 必须是 http(s) 地址，得到：{base_url}")
        if any(h in self.base_url for h in _BLOCKED_HOSTS):
            raise ValueError("base_url 指向被禁止的元数据地址")
        self.timeout_s = timeout_s
        self._transport = transport  # 测试注入（httpx.ASGITransport/MockTransport）

    def invoke(self, request: InvocationRequest) -> TargetObservation:
        import httpx

        try:
            client_kwargs: dict = {"timeout": min(request.timeout_s, self.timeout_s)}
            if self._transport is not None:
                client_kwargs["transport"] = self._transport
            with httpx.Client(**client_kwargs) as client:
                resp = client.post(self.base_url, json={
                    "input": request.input,
                    "variables": request.variables,
                    "metadata": request.metadata,
                })
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # 网络/超时/非 JSON → 单条用例失败
            return TargetObservation(status="error", error=f"{type(exc).__name__}: {exc}")

        status = data.get("status", "success")
        if status not in ("success", "rejected", "error"):
            status = "success"
        return TargetObservation(
            status=status,
            output=str(data.get("output", "")),
            structured_output=data.get("structured_output"),
            tool_calls=list(data.get("tool_calls") or []),
            latency_ms=int(data.get("latency_ms") or 0),
            error=str(data.get("error", "")),
            meta={"http_status": resp.status_code},
        )
