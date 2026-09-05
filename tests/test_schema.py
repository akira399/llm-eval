"""用例集契约测试：真实的 50 条种子用例必须全部通过结构校验。"""
import os

import pytest

from evalkit.schema import Case, CaseError, load_cases

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = os.path.join(_ROOT, "cases", "poke-rag-v0.yaml")

# 种子集的类目标签全集（分组口径见 docs/00-技术方案.md §3.1）
EXPECTED_CATEGORIES = {
    "图鉴", "招式", "特性", "道具", "属性克制", "招式集合", "对战环境", "超范围", "安全", "伤害计算",
}


def test_seed_set_loads_and_is_complete():
    meta, cases = load_cases(CASES)
    assert meta.get("name") == "poke-rag-v0"
    assert len(cases) == 50
    assert {c.category for c in cases} == EXPECTED_CATEGORIES


def test_ids_unique():
    _, cases = load_cases(CASES)
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))


def test_answer_cases_have_key_facts():
    _, cases = load_cases(CASES)
    for case in cases:
        if case.expect == "answer":
            assert case.key_facts, f"{case.id} expect=answer 但没有 key_facts"


def test_difficulty_hard_marked_separately():
    _, cases = load_cases(CASES)
    hard = [c for c in cases if c.difficulty == "hard"]
    assert hard, "种子集应包含已知弱项（hard）用例"
    assert all(c.expect == "answer" for c in hard)
    assert all(c.category == "伤害计算" for c in hard)


def test_reject_cases_have_no_key_facts():
    _, cases = load_cases(CASES)
    for case in cases:
        if case.expect in ("reject", "safe"):
            assert case.key_facts == [], f"{case.id} 拒答/安全类不应有 key_facts"


def test_case_validation_rejects_bad_input():
    with pytest.raises(CaseError):
        Case.from_dict({"id": "x-1", "category": "图鉴", "query": "?", "expect": "maybe"})
    with pytest.raises(CaseError):
        Case.from_dict({"id": "x-1", "category": "图鉴", "query": "?"})  # answer 缺 key_facts
    with pytest.raises(CaseError):
        Case.from_dict({"id": "", "category": "图鉴", "query": "?"})
    with pytest.raises(CaseError):
        Case.from_dict({"id": "x-1", "category": "图鉴", "query": "?", "difficulty": "impossible"})
