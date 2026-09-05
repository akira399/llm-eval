"""裁判用 LLM 客户端（OpenAI 兼容协议，非流式——裁判不需要流式输出）。

与 poke-rag 的 llm.py 同一套协议；temperature 默认 0 保证判分可复现。
"""
from __future__ import annotations

from evalkit.config import load_llm_config


def chat(messages: list[dict], temperature: float = 0.0, model: str | None = None) -> str:
    """单轮补全，返回第一条回复文本（供 Judge 解析 JSON）。"""
    from openai import OpenAI

    cfg = load_llm_config()
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    resp = client.chat.completions.create(
        model=model or cfg["model"],
        messages=messages,
        temperature=temperature,
        stream=False,
    )
    return resp.choices[0].message.content or ""
