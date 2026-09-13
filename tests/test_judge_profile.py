"""JudgeProfile 测试：JSON Schema 校验器、规则评分器、RAG 兼容包装、异常隔离。"""
import pytest

from evalkit.contracts import EvaluationCase, TargetObservation
from evalkit.judge_profile import (
    RagJudgeProfile,
    chat_profile,
    json_profile,
    none_profile,
    rule_boundary,
    rule_fact_coverage,
    rule_field_accuracy,
    rule_json_schema_valid,
    validate_json_schema,
)
from evalkit.schema import TargetResult


def _case(behavior=None, text="q") -> EvaluationCase:
    return EvaluationCase(case_id="c1", input={"text": text},
                          expected_behavior=behavior or {"type": "answer"})


def _obs(output="", structured=None, status="success", meta=None) -> TargetObservation:
    return TargetObservation(status=status, output=output, structured_output=structured, meta=meta or {})


def test_validate_json_schema():
    schema = {"type": "object", "required": ["name", "age"],
              "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
    assert validate_json_schema({"name": "张三", "age": 28}, schema) == []
    assert any("缺少必填字段 age" in i for i in validate_json_schema({"name": "张三"}, schema))
    assert any("期望类型 integer" in i for i in validate_json_schema({"name": "张三", "age": "28"}, schema))
    assert any("期望类型 object" in i for i in validate_json_schema([1, 2], schema))


def test_fact_coverage_rule():
    case = _case({"type": "answer", "facts": ["7 天无理由退款", "运费"]})
    full = rule_fact_coverage(case, _obs("我们支持 7 天无理由退款，且运费自理。"))
    assert full["score"] == 1.0
    partial = rule_fact_coverage(case, _obs("我们支持 7 天无理由退款。"))
    assert partial["score"] == 0.5
    miss = rule_fact_coverage(case, _obs("无法回答"))
    assert miss["score"] == 0.0
    no_facts = rule_fact_coverage(_case({"type": "answer"}), _obs("随便"))
    assert no_facts["score"] is None  # 无要点 → skip，不进均分


def test_boundary_rule():
    reject = _case({"type": "reject"})
    ok = rule_boundary(reject, _obs("抱歉，我只能回答售后问题。"))
    assert ok["score"] == 1.0
    bad = rule_boundary(reject, _obs("今天晴，25 度。"))
    assert bad["score"] == 0.0
    answer = _case({"type": "answer"})
    assert rule_boundary(answer, _obs("正常回答"))["score"] is None  # 非边界用例跳过


def test_json_graders():
    case = _case({"type": "json",
                  "json_schema": {"type": "object", "required": ["name"],
                                  "properties": {"name": {"type": "string"}}},
                  "expect_fields": {"name": "张三"}})
    good = rule_json_schema_valid(case, _obs(structured={"name": "张三"}))
    assert good["score"] == 1.0
    bad = rule_json_schema_valid(case, _obs(structured={}))
    assert bad["score"] == 0.0 and any("缺少必填字段" in i for i in bad["issues"])
    none_out = rule_json_schema_valid(case, _obs(structured=None))
    assert none_out["score"] == 0.0
    field = rule_field_accuracy(case, _obs(structured={"name": "张三"}))
    assert field["score"] == 1.0
    field_bad = rule_field_accuracy(case, _obs(structured={"name": "李四"}))
    assert field_bad["score"] == 0.0 and field_bad["fields"][0]["actual"] == "李四"


def test_profile_exception_isolated_not_fatal():
    def boom(case, obs):
        raise RuntimeError("评分器炸了")

    from evalkit.judge_profile import JudgeProfile

    profile = JudgeProfile(profile_id="t", dimensions=["x"], graders={"x": boom})
    out = profile.judge(_case(), _obs())
    assert out["x"]["score"] is None and "评分器炸了" in out["x"]["error"]


def test_chat_profile_end_to_end_offline():
    profile = chat_profile()
    hit = _case({"type": "answer", "facts": ["退款"]}, text="退款")
    out = profile.judge(hit, _obs("支持 7 天无理由退款"))
    assert out["fact_coverage"]["score"] == 1.0
    assert out["boundary"]["score"] is None
    assert "tone" not in out  # 未启用 LLM 维度


def test_none_profile():
    assert none_profile().judge(_case(), _obs("x")) == {}


def test_rag_profile_reuses_legacy_judge():
    class FakeLLMJudge:
        def judge(self, case, result):
            assert case.id == "dex-001" and case.query == "快龙是什么属性？"
            assert result.answer_text == "龙属性 [1]"
            return {"correctness": {"score": 1.0}, "faithfulness": {"score": None},
                    "format": {"score": 1.0}, "tone": {"score": 2}}

    profile = RagJudgeProfile(judge=FakeLLMJudge())
    rag_meta = {"id": "dex-001", "category": "图鉴", "query": "快龙是什么属性？",
                "expect": "answer", "difficulty": "easy", "key_facts": ["龙属性"], "notes": ""}
    case = EvaluationCase(case_id="dex-001", input={"text": "快龙是什么属性？"},
                          metadata={"rag": rag_meta})
    out = profile.judge(case, _obs("龙属性 [1]", meta={"citations": {"1": "poke:149"}}))
    assert out["correctness"]["score"] == 1.0 and out["tone"]["score"] == 2


def test_rag_profile_reject_case_via_legacy_rules():
    from evalkit.judge import LLMJudge

    # 注入假客户端：拒答类的正确性/格式/忠实度走规则，只有 tone 需要 LLM
    fake = lambda messages: '{"score": 2, "reason": "ok"}'  # noqa: E731
    profile = RagJudgeProfile(judge=LLMJudge(client=fake))
    rag_meta = {"id": "oos-1", "category": "超范围", "query": "天气？",
                "expect": "reject", "difficulty": "easy", "key_facts": [], "notes": ""}
    case = EvaluationCase(case_id="oos-1", input={"text": "天气？"}, metadata={"rag": rag_meta})
    out = profile.judge(case, _obs("知识库中未找到相关信息。"))
    assert out["correctness"]["score"] == 1.0 and out["correctness"]["method"] == "rule"
