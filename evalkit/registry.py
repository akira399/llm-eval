"""目标注册表（Phase 0/1）：平台可评哪些应用，在这里声明。

服务化审查结论：线上不允许客户端传任意 root/URL——目标必须先注册，
评测时只传 target_id。本地阶段同样遵循该纪律（目录/连接细节封装在适配器内）。
"""
from __future__ import annotations

from evalkit.contracts import TargetAdapter

# 元数据目录：列出能力与说明，不实例化（pokerag 依赖本地环境，按需构建）
TARGET_CATALOG: dict[str, dict] = {
    "demo-chat": {"kind": "chat", "description": "规则版售后客服（离线确定性演示）"},
    "demo-json": {"kind": "json", "description": "规则版信息抽取（离线确定性演示）"},
    "pokerag-local": {"kind": "rag", "description": "Poke-RAG 知识库问答（进程内，需本地环境）"},
}


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
    raise KeyError(f"未注册的被测目标：{target_id}（可选：{sorted(TARGET_CATALOG)}）")
