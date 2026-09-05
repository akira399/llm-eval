"""promptfoo 自定义 provider（阶段 0 用 promptfoo 跑同一套被测应用）。

promptfoo 通过 python:// 协议调用本模块的 call_api；内部复用 evalkit 的
PokeRagProvider，保证两条链路（promptfoo / 自研 Runner）测的是同一个东西。
可用环境变量 EVAL_OVERRIDES 传入 JSON 配置覆盖（与 run_eval.py --override 等价）。
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_PROVIDER = None


def _get_provider():
    global _PROVIDER
    if _PROVIDER is None:
        from evalkit.providers.pokerag import PokeRagProvider

        overrides = None
        raw = os.environ.get("EVAL_OVERRIDES")
        if raw:
            overrides = json.loads(raw)
        _PROVIDER = PokeRagProvider(overrides=overrides)
    return _PROVIDER


def call_api(prompt: str, options=None, context=None) -> dict:
    """promptfoo 约定：prompt 是渲染后的查询，返回 {"output": ...}。"""
    result = _get_provider().answer(prompt)
    if result.error:
        return {"output": "", "error": result.error}
    output = result.answer_text if not result.rejected else f"【拒答】{result.answer_text}"
    return {
        "output": output,
        "metadata": {
            "latency_ms": result.latency_ms,
            "citations": result.citations,
            "rejected": result.rejected,
            "citations_ok": result.citations_ok,
        },
    }
