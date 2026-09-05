"""被测适配器测试：配置快照注入与元数据（轻量，不触发 LLM/检索）。"""
import json
import os
import sys

import pytest

from evalkit.config import poke_rag_root
from evalkit.providers.pokerag import PokeRagProvider, _deep_merge

pytest.importorskip("yaml")  # cases 依赖已保证，这里只是防御


def test_deep_merge_nested():
    base = {"llm": {"model": "m1", "temperature": 0.2}, "rag": {"top_k_answer": 5}}
    merged = _deep_merge(base, {"rag": {"top_k_answer": 8}})
    assert merged["rag"]["top_k_answer"] == 8
    assert merged["llm"] == {"model": "m1", "temperature": 0.2}  # 其他键不受影响
    assert base["rag"]["top_k_answer"] == 5  # 原配置不被改动


def test_provider_requires_existing_root():
    with pytest.raises(FileNotFoundError):
        PokeRagProvider(root="Z:/no/such/dir")


def test_provider_overrides_via_temp_config():
    """overrides 生效：config 模块被重定向到合并后的临时文件，用户配置不动。"""
    root = poke_rag_root()
    if not os.path.isdir(root):
        pytest.skip("poke-rag 不在默认位置")
    provider = PokeRagProvider(root=root, overrides={"rag": {"top_k_answer": 8}})
    loaded = provider.effective_config()
    assert loaded["rag"]["top_k_answer"] == 8
    # 其他配置保持 poke-rag 本地值（llm 节原样带过来）
    assert loaded["llm"]["model"]  # 用户已配置模型
    # 临时配置文件已重定向
    assert provider._tmp_config and os.path.exists(provider._tmp_config)
    assert str(provider._tmp_config) != os.path.join(root, "config.local.json")


def test_poke_rag_on_sys_path():
    root = poke_rag_root()
    if not os.path.isdir(root):
        pytest.skip("poke-rag 不在默认位置")
    provider = PokeRagProvider(root=root)
    assert root in sys.path
    assert provider.effective_config().get("rag") is not None


def test_effective_config_masks_api_key():
    """运行元数据里的 api_key 必须脱敏（元数据会落盘 summary.json）。"""
    root = poke_rag_root()
    if not os.path.isdir(root):
        pytest.skip("poke-rag 不在默认位置")
    provider = PokeRagProvider(root=root)
    cfg = provider.effective_config()
    key = cfg.get("llm", {}).get("api_key") or ""
    assert "****" in key or key == ""
    assert "sk-4cff" not in json.dumps(cfg)  # 任何已知明文 key 形态都不允许出现
