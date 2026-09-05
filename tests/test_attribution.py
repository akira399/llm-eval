"""失败归因测试：卡片解析、两级判定、整批归因。"""
from evalkit.attribution import attribute, attribute_run, resolve_card
from evalkit.schema import Case


def _case(**kw) -> Case:
    defaults = dict(id="t-1", category="图鉴", query="q", key_facts=["f"], expect_card_en="dragonite")
    defaults.update(kw)
    return Case(**defaults)


def _record(case: Case, target: dict, judge_correctness=None) -> dict:
    return {"case": case.to_dict(), "target": target,
            "judge": {"correctness": {"score": judge_correctness}}}


def test_resolve_card_passthrough_and_title():
    assert resolve_card("rule:move_query") == "rule:move_query"
    assert resolve_card("dragonite") == "poke:149"
    assert resolve_card("dragon-type matchups") == "type:dragon"
    assert resolve_card("gen9ou") == "meta:gen9ou"
    assert resolve_card("no-such-card") is None
    assert resolve_card("") is None


def test_attribute_error_and_reject():
    case = _case(expect="reject", key_facts=[])
    out = attribute(case, _record(case, {"error": "RuntimeError: x"}), None)
    assert out["layer"] == "运行"
    out = attribute(case, _record(case, {"rejected": True, "answer_text": "未找到"}), None)
    assert out["layer"] == "检索层" and out["label"] == "拒答闸触发"


def test_attribute_generation_vs_retrieval():
    case = _case()
    # 期望卡片在引用里 → 生成层
    rec = _record(case, {"citations": {"1": "poke:149"}, "answer_text": "答错的内容"}, 0.0)
    out = attribute(case, rec, "poke:149")
    assert out["layer"] == "生成层"
    # 期望卡片不在引用里 → 检索/组装层
    rec = _record(case, {"citations": {"1": "poke:25"}, "answer_text": "未找到相关信息"}, 0.0)
    out = attribute(case, rec, "poke:149")
    assert out["layer"] == "检索/组装层"
    # 通过 → 不标层
    out = attribute(case, _record(case, {"citations": {"1": "poke:149"}, "answer_text": "正确"}, 1.0), "poke:149")
    assert out["label"] == "通过"


def test_attribute_without_expected_card():
    case = _case(expect_card_en="")
    rec = _record(case, {"citations": {"1": "poke:25"}, "answer_text": "半对"}, 0.0)
    out = attribute(case, rec, None)
    assert out["label"] == "未指定期望卡片"


def test_attribute_run_filters_passes():
    good_case = _case(id="g-1")
    bad_case = _case(id="b-1", expect_card_en="dragonite")
    records = [
        _record(good_case, {"citations": {"1": "poke:149"}, "answer_text": "正确"}, 1.0),
        _record(bad_case, {"rejected": True, "answer_text": "未找到"}),
    ]
    result = attribute_run(records)
    assert set(result.keys()) == {"b-1"}  # 通过的不进结果
    assert result["b-1"]["layer"] == "检索层"
