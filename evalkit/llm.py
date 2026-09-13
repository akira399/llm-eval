"""裁判用 LLM 客户端（OpenAI 兼容协议，非流式——裁判不需要流式输出）。

与 poke-rag 的 llm.py 同一套协议；temperature 默认 0 保证判分可复现。
"""
from __future__ import annotations

from evalkit.config import load_llm_config


def chat(messages: list[dict], temperature: float = 0.0, model: str | None = None,
         on_usage=None, timeout_s: int = 120) -> str:
    """单轮补全，返回第一条回复文本（供 Judge 解析 JSON）。

    on_usage：可选回调，收到 {"model","input_tokens","output_tokens"}——
    成本账本（Phase 2.5）的数据源头；超时防止单次裁判调用拖垮整批评测。
    """
    from openai import OpenAI

    cfg = load_llm_config()
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=timeout_s)
    resp = client.chat.completions.create(
        model=model or cfg["model"],
        messages=messages,
        temperature=temperature,
        stream=False,
    )
    if on_usage is not None and getattr(resp, "usage", None):
        on_usage({
            "model": model or cfg["model"],
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        })
    return resp.choices[0].message.content or ""
