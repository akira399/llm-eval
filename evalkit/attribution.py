"""两级失败归因（方案 §6）：失败出在检索层还是生成层？

对每个低分/异常用例，按顺序判定：
1. 运行错误 → 环境问题；
2. 拒答 → 检索层（无有效召回或低于置信度闸）；
3. 期望卡片在引用里 → 生成层（检索给了素材，回答没答好）；
4. 期望卡片不在引用里 → 检索/组装层（需人工区分是没检到还是组装丢弃——
   两者对外症状相同，v1 的 Meta 失败即组装丢弃的实例）。

期望卡片由用例的 expect_card_en 指定（title_en 或直接 card_id），可省略。
"""
from __future__ import annotations

import json
import os

from evalkit.config import poke_rag_root
from evalkit.schema import Case


def resolve_card(value: str) -> str | None:
    """title_en（大小写不敏感）→ card_id；带 ':' 的值视为 card_id 直接透传。"""
    if not value:
        return None
    if ":" in value:
        return value
    root = os.path.join(poke_rag_root(), "data", "cards")
    if not os.path.isdir(root):
        return None
    for name in os.listdir(root):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(root, name), encoding="utf-8") as f:
            for line in f:
                card = json.loads(line)
                if card.get("title_en", "").lower() == value.lower():
                    return card["card_id"]
    return None


def attribute(case: Case, record: dict, expected_card_id: str | None) -> dict:
    target = record["target"]
    judge = record.get("judge") or {}
    correctness = (judge.get("correctness") or {}).get("score")
    failed = bool(target.get("error")) or target.get("rejected") or (
        correctness is not None and correctness < 0.5
    )
    out: dict = {"case_id": case.id, "correctness": correctness, "failed": failed}
    if target.get("error"):
        out.update(layer="运行", label="运行出错", detail=target["error"])
        return out
    if target.get("rejected"):
        out.update(layer="检索层", label="拒答闸触发",
                   detail="无有效召回或置信度低于阈值，未进入生成")
        return out
    if not failed:
        out.update(layer="-", label="通过", detail="")
        return out
    if not expected_card_id:
        out.update(layer="?", label="未指定期望卡片",
                   detail="用例缺少 expect_card_en，无法自动定位（对照引用卡片人工归因）")
        return out
    cited = set((target.get("citations") or {}).values())
    if expected_card_id in cited:
        out.update(layer="生成层", label="检索命中但回答未覆盖要点",
                   detail=f"期望卡片 {expected_card_id} 已进上下文，检查提示词约束/模型能力")
    else:
        out.update(layer="检索/组装层", label="期望卡片未出现在引用中",
                   detail=f"期望 {expected_card_id}，实际引用 {sorted(cited)[:5]}；"
                          "需区分检索未命中 vs 组装丢弃")
    return out


def attribute_run(records: list[dict], cases: dict[str, Case] | None = None) -> dict:
    """整批归因：{case_id: 归因}，只处理失败与异常用例。

    cases 可传当前用例集（按 id 匹配）：运行记录内嵌的是运行当时的用例快照，
    归因老记录时用新元数据（如后补的 expect_card_en）。
    """
    by_id = {r["case"]["id"]: r for r in records}
    result: dict = {}
    for case_id, record in by_id.items():
        case = (cases or {}).get(case_id) or Case.from_dict(record["case"])
        expected = resolve_card(case.expect_card_en) if case.expect_card_en else None
        if case.expect_card_en and expected is None:
            result[case_id] = {"case_id": case_id, "failed": True, "layer": "?",
                               "label": "期望卡片无法解析", "detail": case.expect_card_en}
            continue
        attr = attribute(case, record, expected)
        if attr["failed"] or attr["label"] not in ("通过",):
            result[case_id] = attr
    return result
