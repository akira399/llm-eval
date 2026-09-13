"""评测平台的配置：裁判模型配置 + 被测系统定位。

裁判模型默认复用 poke-rag 的 config.local.json（同一把 DeepSeek key，
评测平台不单独配第二把钥匙）；如 llm-eval/config.local.json 存在则优先。
环境变量 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 优先级最高。
"""
from __future__ import annotations

import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 裁判与被测系统可以同模型；裁判要稳定，温度固定 0（除非本地配置显式给）
JUDGE_TEMPERATURE = 0.0


def poke_rag_root() -> str:
    """优先级：环境变量 > 用户配置文件 targets.local.json > 默认兄弟目录。"""
    root = os.environ.get("POKE_RAG_ROOT")
    if not root:
        try:
            from evalkit.targets_store import pokerag_root_override

            root = pokerag_root_override()
        except Exception:
            root = ""
    if not root:
        root = os.path.join(os.path.dirname(_ROOT), "poke-rag")
    return os.path.abspath(root)


def load_llm_config() -> dict:
    """裁判用 LLM 配置：llm-eval 本地配置 > poke-rag 配置 > 环境变量覆盖。"""
    cfg = {"base_url": "", "api_key": "", "model": "", "temperature": JUDGE_TEMPERATURE}
    for path in (
        os.path.join(_ROOT, "config.local.json"),
        os.path.join(poke_rag_root(), "config.local.json"),
    ):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            user = json.load(f).get("llm", {})
        for key in ("base_url", "api_key", "model"):
            if user.get(key):
                cfg[key] = user[key]
        if user.get("temperature") is not None:
            cfg["temperature"] = user["temperature"]
        break
    cfg["base_url"] = os.environ.get("LLM_BASE_URL", cfg["base_url"])
    cfg["api_key"] = os.environ.get("LLM_API_KEY", cfg["api_key"])
    cfg["model"] = os.environ.get("LLM_MODEL", cfg["model"])
    return cfg
