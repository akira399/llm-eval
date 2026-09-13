"""用户自定义目标与评测集导入测试（Phase 2.6：让用户接入自己的项目）。"""
import json
import os

import pytest

from evalkit import targets_store
from evalkit.registry import build_target, effective_catalog
from evalkit.service import EvalService, ServiceError

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def env_root(tmp_path, monkeypatch):
    """独立 root：targets.local.json 与 suites 都落在临时目录。"""
    monkeypatch.setenv("LLM_EVAL_ROOT", str(tmp_path))
    os.makedirs(os.path.join(str(tmp_path), "suites"), exist_ok=True)
    shutil_copy = __import__("shutil").copy
    shutil_copy(os.path.join(_REPO, "suites", "demo-chat.yaml"),
                os.path.join(str(tmp_path), "suites", "demo-chat.yaml"))
    return tmp_path


def test_http_target_crud(env_root):
    added = targets_store.add_http_target("my-app", "http://127.0.0.1:9000/invoke", "我的应用")
    assert added["target_id"] == "my-app"
    assert "my-app" in effective_catalog()
    adapter = build_target("my-app")
    assert adapter.base_url == "http://127.0.0.1:9000/invoke"
    targets_store.remove_http_target("my-app")
    assert "my-app" not in effective_catalog()
    with pytest.raises(ValueError):
        targets_store.remove_http_target("my-app")


def test_http_target_validation(env_root):
    with pytest.raises(ValueError):  # 非法 id
        targets_store.add_http_target("My App!", "http://x/invoke")
    with pytest.raises(ValueError):  # 与内置冲突
        targets_store.add_http_target("demo-chat", "http://x/invoke")
    with pytest.raises(ValueError):  # 非 http(s)
        targets_store.add_http_target("ok-id", "ftp://x")
    with pytest.raises(ValueError):  # 云元数据地址
        targets_store.add_http_target("ok-id", "http://169.254.169.254/latest")
    with pytest.raises(ValueError):  # 重复注册
        targets_store.add_http_target("dup", "http://x/invoke")
        targets_store.add_http_target("dup", "http://x/invoke")


def test_pokerag_root_override(env_root, monkeypatch):
    monkeypatch.delenv("POKE_RAG_ROOT", raising=False)
    from evalkit.config import poke_rag_root

    import evalkit.config as cfg_mod

    default = os.path.abspath(os.path.join(os.path.dirname(cfg_mod.__file__), "..", "..", "poke-rag"))
    assert poke_rag_root() == default  # 未配置 → 默认兄弟目录
    targets_store.save_user_config(pokerag_root=str(env_root))
    assert poke_rag_root() == str(env_root)  # 用户配置生效
    monkeypatch.setenv("POKE_RAG_ROOT", "E:/env-wins")  # 环境变量优先级更高
    assert poke_rag_root() == os.path.abspath("E:/env-wins")


def test_corrupt_config_does_not_crash(env_root):
    with open(targets_store.config_path(), "w", encoding="utf-8") as f:
        f.write("{broken json")
    assert targets_store.load_user_config() == {"pokerag_root": "", "http_targets": []}


def test_save_suite_validates_and_persists(env_root):
    service = EvalService(root=str(env_root))
    content = (
        "meta: {name: mine, format: generic}\n"
        "cases:\n"
        "  - {case_id: c1, input: {text: q}, expected_behavior: {type: answer, facts: [f]}}\n")
    result = service.save_suite("mine", content)
    assert result["case_count"] == 1
    assert any(s["suite_id"] == "mine" for s in service.list_suites())

    with pytest.raises(ServiceError) as ei:  # 重名不覆盖
        service.save_suite("mine", content)
    assert ei.value.code == "ALREADY_EXISTS"
    assert service.save_suite("mine", content, overwrite=True)["case_count"] == 1

    bad = "meta: {name: bad, format: generic}\ncases: [{case_id: b1}]"
    with pytest.raises(ServiceError) as ei2:  # 校验失败 → 回滚
        service.save_suite("bad", bad)
    assert ei2.value.code == "INVALID_ARGUMENT"
    assert not os.path.exists(os.path.join(str(env_root), "suites", "bad.yaml"))


def test_check_target_live(env_root):
    service = EvalService(root=str(env_root))
    result = service.check_target("demo-chat")
    assert result["status"] == "success" and "退款" in result["output"]
