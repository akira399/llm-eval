"""目标注册表（Phase 0/1；Phase 2.6 起支持用户自定义目标）。

服务化审查结论：线上不允许客户端传任意 root/URL——目标必须先注册，
评测时只传 target_id。本地阶段同样遵循该纪律（目录/连接细节封装在适配器内）。

有效目标 = 内置目标 + 用户自定义（evalkit/targets_store.py 维护的
targets.local.json：pokerag 本地目录覆盖 + 用户 HTTP 目标）。
"""
from __future__ import annotations

from evalkit.contracts import TargetAdapter
from evalkit.targets_store import build_user_http_target, load_user_config

# 内置目标：连接细节在平台代码里，用户不得同名覆盖
BUILTIN_TARGET_CATALOG: dict[str, dict] = {
    "demo-chat": {"kind": "chat", "description": "规则版售后客服（离线确定性演示）"},
    "demo-json": {"kind": "json", "description": "规则版信息抽取（离线确定性演示）"},
    "pokerag-local": {"kind": "rag", "description": "Poke-RAG 知识库问答（进程内，需本地环境）"},
    "demo-chat-http": {"kind": "chat", "description": "演示客服（HTTP JSON 连接器，"
                       "base_url 由 DEMO_CHAT_HTTP_URL 指定，默认 127.0.0.1:8766/invoke）"},
}
BUILTIN_TARGET_IDS = frozenset(BUILTIN_TARGET_CATALOG)


def effective_catalog() -> dict[str, dict]:
    """内置 + 用户自定义（每次调用重读配置，工作台改完即生效）。"""
    catalog = dict(BUILTIN_TARGET_CATALOG)
    for t in load_user_config()["http_targets"]:
        tid = t.get("target_id")
        if tid and tid not in catalog:
            catalog[tid] = {"kind": "chat", "user_defined": True,
                            "description": t.get("description", "")}
    return catalog


def catalog_for(target_id: str) -> dict | None:
    return effective_catalog().get(target_id)


# 兼容旧引用（脚本/工作台遍历用）；动态场景请改用 effective_catalog()
def TARGET_CATALOG() -> dict[str, dict]:
    return effective_catalog()


def build_target(target_id: str) -> TargetAdapter:
    """按 target_id 构建适配器实例（未知 id 直接拒绝，不做动态解析）。"""
    if target_id == "demo-chat":
        from evalkit.demo import DemoChatAdapter

        return DemoChatAdapter()
    if target_id == "demo-json":
        from evalkit.demo import DemoJsonAdapter

        return DemoJsonAdapter()
    if target_id == "pokerag-local":
        from evalkit.providers.pokerag import PokeRagAdapter

        return PokeRagAdapter()
    if target_id == "demo-chat-http":
        import os

        from evalkit.providers.http_json import HttpJsonAdapter

        base_url = os.environ.get("DEMO_CHAT_HTTP_URL", "http://127.0.0.1:8766/invoke")
        return HttpJsonAdapter(base_url=base_url, target_id=target_id)
    try:  # 用户自定义 HTTP 目标
        return build_user_http_target(target_id)
    except KeyError:
        raise KeyError(f"未注册的被测目标：{target_id}（可选：{sorted(effective_catalog())}）") from None
