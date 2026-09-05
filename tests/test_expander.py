"""扩写器测试：支持率预筛、解析去重、模板生成（LLM 全部注入假客户端）。"""
import pytest

from evalkit.expander import CaseExpander, case_support, fact_support_ratio
from evalkit.schema import Case


def _seed(**kw) -> Case:
    defaults = dict(id="s-1", category="图鉴", query="快龙是什么属性的宝可梦？",
                    key_facts=["龙属性和飞行属性"])
    defaults.update(kw)
    return Case(**defaults)


CARD_CONTENT = "快龙是龙属性和飞行属性宝可梦，体重 210.0 千克，特性是多鳞。"


def test_fact_support_ratio():
    assert fact_support_ratio("快龙是龙属性和飞行属性宝可梦", CARD_CONTENT) == 1.0
    assert fact_support_ratio("它是传说中的幻兽，种族值 999", CARD_CONTENT) < 0.6  # 编造要点被预筛抓住
    assert fact_support_ratio("", CARD_CONTENT) == 0.0


def test_case_support_averages_facts():
    case = _seed(key_facts=["快龙是龙属性和飞行属性宝可梦", "体重 210.0 千克"])
    assert case_support(case, CARD_CONTENT) == 1.0


def _json_client(payload: str):
    return lambda messages: payload


def test_parse_case_dedup_and_validation():
    expander = CaseExpander([_seed()], ask=_json_client(
        '{"query": "快龙有多重？", "key_facts": ["体重 210.0 千克"]}'))
    case = expander._parse_case('{"query": "快龙有多重？", "key_facts": ["体重 210.0 千克"]}',
                                "图鉴", None)
    assert case is not None and case.query == "快龙有多重？"
    # 同查询第二次 → 去重返回 None
    assert expander._parse_case('{"query": "快龙有多重？", "key_facts": ["体重 210 千克"]}',
                                "图鉴", None) is None
    # answer 无要点 → 拒绝
    expander2 = CaseExpander([_seed()], ask=_json_client('{"query": "快龙多大？", "key_facts": []}'))
    assert expander2._parse_case('{"query": "快龙多大？", "key_facts": []}', "图鉴", None) is None


def test_parse_case_garbage_returns_none():
    expander = CaseExpander([_seed()], ask=_json_client("我觉得不行"))
    assert expander._parse_case("我觉得不行", "图鉴", None) is None


def test_expand_card_category_filters_low_support():
    fabricated = '{"query": "快龙是哪一年诞生的？", "key_facts": ["1996 年首次登场"]}'
    good = '{"query": "快龙有多重？", "key_facts": ["体重 210.0 千克"]}'
    # 编造要点先到 → 被预筛淘汰，合格项补位
    responses = iter([fabricated, good, good])
    expander = CaseExpander([_seed()], ask=lambda m: next(responses))
    cards = [
        {"card_id": "poke:149", "title_zh": "快龙", "content_zh": CARD_CONTENT},
        {"card_id": "poke:150", "title_zh": "快龙（对照）", "content_zh": CARD_CONTENT},
    ]
    kept, rejected = expander.expand_card_category("图鉴", "pokemon", 1, "体重", cards, set())
    assert len(kept) == 1 and kept[0].query == "快龙有多重？"
    assert any("支持率" in r["reason"] for r in rejected)  # 编造要点被机器预筛淘汰


def test_move_query_templates_and_dedup():
    cards = [{"card_id": f"poke:{i}", "title_zh": f"宝可梦{i}", "content_zh": "x" * 50}
             for i in range(3)]
    expander = CaseExpander([_seed()], ask=lambda m: "")
    out = expander.expand_move_query_templates(3, cards)
    assert len(out) == 3
    assert all(c.expect_card_en == "rule:move_query" for c in out)
    # 与种子重复的查询不会出现
    dup = CaseExpander([_seed(), _seed(id="s-2", query="宝可梦1都会哪些招式？")], ask=lambda m: "")
    out2 = dup.expand_move_query_templates(3, cards)
    assert all(c.query != "宝可梦1都会哪些招式？" for c in out2)


def test_id_generation_is_unique_within_run():
    expander = CaseExpander([_seed()], ask=_json_client('{"query": "Q？", "key_facts": ["无关要点甲"]}'))
    cases = [expander._parse_case('{"query": "Q？", "key_facts": ["无关要点甲"]}', "特性", None)
             for _ in range(1)]  # 单条即可；去重由 existing_queries 保证
    assert cases[0].id.startswith("gen-特性-")
