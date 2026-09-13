"""通用契约测试：EvaluationCase 校验、加载器、新旧契约互通、演示适配器。"""
import os

import pytest

from evalkit.contracts import (
    ContractError,
    EvaluationCase,
    InvocationRequest,
    TargetObservation,
    evaluation_case_from_rag,
    load_evaluation_cases,
    observation_from_target_result,
)
from evalkit.demo import DemoChatAdapter, DemoJsonAdapter
from evalkit.registry import build_target
from evalkit.schema import Case, TargetResult

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_evaluation_case_validation():
    ok = EvaluationCase.from_dict({"case_id": "c1", "input": {"text": "hi"},
                                   "expected_behavior": {"type": "answer", "facts": ["a"]}})
    assert ok.case_id == "c1" and ok.expected_behavior["type"] == "answer"
    with pytest.raises(ContractError):
        EvaluationCase.from_dict({"input": {"text": "hi"}})                      # 缺 case_id
    with pytest.raises(ContractError):
        EvaluationCase.from_dict({"case_id": "c2"})                             # 缺 input
    with pytest.raises(ContractError):
        EvaluationCase.from_dict({"case_id": "c3", "input": {"text": "x"},
                                  "expected_behavior": {"type": "maybe"}})      # 未知 type
    with pytest.raises(ContractError):
        EvaluationCase.from_dict({"case_id": "c4", "input": {"text": "x"},
                                  "expected_behavior": {"type": "json"}})       # json 缺 schema
    with pytest.raises(ContractError):
        EvaluationCase.from_dict({"case_id": "c5", "input": {"text": "x"},
                                  "difficulty": "impossible"})


def test_invocation_request_text():
    assert InvocationRequest(input={"text": "你好"}).text() == "你好"
    req = InvocationRequest(input={"messages": [{"role": "system", "content": "s"},
                                                {"role": "user", "content": "问"}]})
    assert req.text() == "问"


def test_load_demo_suites():
    _, chat = load_evaluation_cases(os.path.join(_ROOT, "suites", "demo-chat.yaml"))
    assert len(chat) == 6
    assert {c.case_id[:5] for c in chat} == {"chat-"}
    _, js = load_evaluation_cases(os.path.join(_ROOT, "suites", "demo-json.yaml"))
    assert len(js) == 5
    assert all(c.expected_behavior["type"] == "json" for c in js)


def test_load_suite_rejects_duplicates(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(
        "meta: {name: x, format: generic}\n"
        "cases:\n"
        "  - {case_id: a, input: {text: q}}\n"
        "  - {case_id: a, input: {text: q}}\n", encoding="utf-8")
    with pytest.raises(ContractError):
        load_evaluation_cases(str(path))


def test_rag_case_round_trip():
    rag = Case(id="dex-001", category="图鉴", query="快龙是什么属性？",
               key_facts=["龙属性"], expect_card_en="dragonite")
    generic = evaluation_case_from_rag(rag)
    assert generic.case_id == "dex-001"
    assert generic.input == {"text": "快龙是什么属性？"}
    assert generic.expected_behavior == {"type": "answer", "facts": ["龙属性"]}
    assert generic.metadata["rag"]["id"] == "dex-001"


def test_target_result_to_observation():
    tr = TargetResult(answer_text="答 [1]", citations={"1": "poke:149"},
                      cited_cards=[{"card_id": "poke:149"}], latency_ms=7)
    obs = observation_from_target_result(tr)
    assert obs.status == "success" and obs.output == "答 [1]"
    assert obs.meta["citations"] == {"1": "poke:149"}
    assert obs.legacy_fields()["answer_text"] == "答 [1]"


def test_demo_chat_adapter():
    adapter = DemoChatAdapter()
    ok = adapter.invoke(InvocationRequest(input={"text": "退款政策是什么"}))
    assert ok.status == "success" and "7 天无理由退款" in ok.output
    refused = adapter.invoke(InvocationRequest(input={"text": "今天天气如何"}))
    assert refused.status == "success" and "抱歉" in refused.output  # 礼貌拒答（success 承认边界）
    attack = adapter.invoke(InvocationRequest(input={"text": "我要攻击你"}))
    assert attack.status == "rejected"


def test_demo_json_adapter():
    adapter = DemoJsonAdapter()
    out = adapter.invoke(InvocationRequest(input={"text": "姓名：张三，年龄28岁"}))
    assert out.structured_output == {"name": "张三", "age": 28}
    missing = adapter.invoke(InvocationRequest(input={"text": "姓名：李四"}))
    assert missing.structured_output == {"name": "李四"}  # 缺 age → 评分器应抓出
    empty = adapter.invoke(InvocationRequest(input={"text": "今天天气不错"}))
    assert empty.structured_output == {}


def test_registry_rejects_unknown_target():
    with pytest.raises(KeyError):
        build_target("no-such-target")
    adapter = build_target("demo-chat")
    assert adapter.target_id == "demo-chat"
