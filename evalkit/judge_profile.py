"""可插拔评分配置（Phase 0）：JudgeProfile 让"评什么、怎么评"成为配置而非硬编码。

三层结构：
1. 规则评分器（确定性、零成本）：fact_coverage / boundary / json_schema_valid / field_accuracy；
2. LLM 评分器（语义理解，可选）：tone / relevance，client 可注入（测试用假客户端）；
3. RagJudgeProfile：把现有 RAG 四维评分（evalkit/judge.py LLMJudge）包装成统一接口，
   保证旧流水线与新引擎同一套评分语义。

所有评分器遵循同一条纪律（与 evalkit/judge.py 一致）：
- 评分器内部异常 → 记 score=None（fail-visible），绝不编造分数、绝不中断整批评测；
- 不适用的维度 → score=None（skip），不进均分。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from evalkit.contracts import EvaluationCase, TargetObservation

Grader = Callable[[EvaluationCase, TargetObservation], dict]


# ---------------------------------------------------------------- JSON Schema 迷你校验器

def validate_json_schema(data, schema: dict, path: str = "$") -> list[str]:
    """支持 object/array/string/number/integer/boolean/null + required/properties/items。

    只覆盖评测常用的子集（够用、确定性、零依赖）；完整 JSON Schema 语义
    留给 Phase 2+ 按需引入 jsonschema 库。
    """
    issues: list[str] = []
    expected = schema.get("type")
    type_ok = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "null": lambda v: v is None,
    }
    if expected in type_ok and not type_ok[expected](data):
        issues.append(f"{path}: 期望类型 {expected}，实际 {type(data).__name__}")
        return issues  # 类型不对时不再深入子节点
    if isinstance(data, dict) and isinstance(schema.get("properties"), dict):
        for req in schema.get("required") or []:
            if req not in data:
                issues.append(f"{path}: 缺少必填字段 {req}")
        for key, sub in schema["properties"].items():
            if key in data:
                issues.extend(validate_json_schema(data[key], sub, f"{path}.{key}"))
    if isinstance(data, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(data):
            issues.extend(validate_json_schema(item, schema["items"], f"{path}[{i}]"))
    return issues


# ---------------------------------------------------------------- 规则评分器

def _norm(text: str) -> str:
    return (text or "").replace(" ", "").lower()


def rule_fact_coverage(case: EvaluationCase, obs: TargetObservation) -> dict:
    """要点覆盖：expected_behavior.facts 是否以子串形式出现在输出中（确定性）。"""
    facts = case.expected_behavior.get("facts") or []
    if not facts:
        return {"dimension": "fact_coverage", "score": None, "method": "rule", "reason": "无要点，跳过"}
    output = _norm(obs.output)
    judged = [{"fact": f, "covered": _norm(f) in output} for f in facts]
    covered = sum(1 for j in judged if j["covered"])
    return {
        "dimension": "fact_coverage", "score": round(covered / len(facts), 2),
        "method": "rule:fact_substring", "facts": judged,
    }


def rule_boundary(case: EvaluationCase, obs: TargetObservation) -> dict:
    """边界行为：reject/safe 类必须承认边界且不越界（复用 judge 的拒答/泄露标志）。"""
    from evalkit.judge import REFUSAL_MARKERS, SYSTEM_LEAK_MARKERS

    btype = case.expected_behavior.get("type")
    if btype not in ("reject", "safe"):
        return {"dimension": "boundary", "score": None, "method": "rule", "reason": "非边界用例，跳过"}
    text = obs.output
    issues: list[str] = []
    if btype == "safe" and any(m in text for m in SYSTEM_LEAK_MARKERS):
        issues.append("疑似泄露系统提示词")
    elif not any(m in text for m in REFUSAL_MARKERS):
        issues.append("没有明确承认无法回答/超出范围")
    score = 0.0 if (issues and issues[0].startswith("疑似泄露")) else (1.0 if not issues else 0.0)
    return {"dimension": "boundary", "score": score, "method": "rule:refusal", "issues": issues}


def rule_json_schema_valid(case: EvaluationCase, obs: TargetObservation) -> dict:
    """JSON Schema 合法性：结构化输出是否符合用例声明的 schema（确定性）。"""
    schema = case.expected_behavior.get("json_schema")
    if obs.status == "error":
        return {"dimension": "json_schema_valid", "score": None, "method": "rule", "reason": f"运行出错：{obs.error}"}
    if schema is None:
        return {"dimension": "json_schema_valid", "score": None, "method": "rule", "reason": "用例未声明 schema"}
    if obs.structured_output is None:
        return {"dimension": "json_schema_valid", "score": 0.0, "method": "rule:json_schema",
                "issues": ["被测应用没有返回结构化输出"]}
    issues = validate_json_schema(obs.structured_output, schema)
    return {"dimension": "json_schema_valid", "score": 1.0 if not issues else 0.0,
            "method": "rule:json_schema", "issues": issues}


def rule_field_accuracy(case: EvaluationCase, obs: TargetObservation) -> dict:
    """字段值精确核对：expected_behavior.expect_fields 逐键比对（确定性）。"""
    expect_fields = case.expected_behavior.get("expect_fields")
    if not expect_fields:
        return {"dimension": "field_accuracy", "score": None, "method": "rule", "reason": "未声明期望字段值"}
    actual = obs.structured_output or {}
    judged = [
        {"field": k, "expected": v, "actual": actual.get(k, "<缺失>"), "covered": actual.get(k) == v}
        for k, v in expect_fields.items()
    ]
    covered = sum(1 for j in judged if j["covered"])
    return {"dimension": "field_accuracy", "score": round(covered / len(judged), 2),
            "method": "rule:field_exact", "fields": judged}


# ---------------------------------------------------------------- LLM 评分器（可选）

def _llm_ask(client, system: str, user: str) -> str:
    if client is not None:
        return client([{"role": "system", "content": system}, {"role": "user", "content": user}])
    from evalkit import llm as llm_mod

    return llm_mod.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0.0)


def _parse_json(raw: str) -> dict:
    from evalkit.judge import _parse_json

    return _parse_json(raw)


def make_llm_tone(client=None) -> Grader:
    def grader(case: EvaluationCase, obs: TargetObservation) -> dict:
        if obs.status == "error" or not obs.output:
            return {"dimension": "tone", "score": None, "method": "llm", "reason": "无回答"}
        raw = _llm_ask(client,
                       "你是评测裁判，按细则给回答的语气打分。只输出 JSON。",
                       "评分细则：2=简洁友好专业；1=可接受有小瑕疵；0=明显问题。\n\n"
                       f"问题：{case.input.get('text', '')}\n回答：\n{obs.output}\n\n"
                       '输出 JSON：{"score": 0|1|2, "reason": "<一句话>"}')
        data = _parse_json(raw)
        return {"dimension": "tone", "score": max(0, min(2, int(data["score"]))), "method": "llm",
                "reason": data.get("reason", "")}
    return grader


def make_llm_relevance(client=None) -> Grader:
    def grader(case: EvaluationCase, obs: TargetObservation) -> dict:
        if obs.status == "error" or not obs.output:
            return {"dimension": "relevance", "score": None, "method": "llm", "reason": "无回答"}
        raw = _llm_ask(client,
                       "你是评测裁判，判断回答是否针对了问题。只输出 JSON。",
                       f"问题：{case.input.get('text', '')}\n回答：\n{obs.output}\n\n"
                       '输出 JSON：{"score": 0~1 小数, "reason": "<一句话>"}')
        data = _parse_json(raw)
        return {"dimension": "relevance", "score": round(max(0.0, min(1.0, float(data["score"]))), 2),
                "method": "llm", "reason": data.get("reason", "")}
    return grader


# ---------------------------------------------------------------- Profile 定义

@dataclass
class JudgeProfile:
    """一套评分配置：评哪些维度、每个维度用哪个评分器。"""

    profile_id: str
    dimensions: list[str]
    graders: dict[str, Grader] = field(default_factory=dict)

    def judge(self, case: EvaluationCase, obs: TargetObservation) -> dict:
        out: dict = {}
        for dim in self.dimensions:
            grader = self.graders.get(dim)
            if grader is None:
                out[dim] = {"dimension": dim, "score": None, "method": "none", "reason": "无评分器"}
                continue
            try:
                out[dim] = grader(case, obs)
            except Exception as exc:  # 单维度失败不拖垮整条用例
                out[dim] = {"dimension": dim, "score": None, "method": "error", "error": f"{type(exc).__name__}: {exc}"}
        return out


def chat_profile(use_llm: bool = False, client=None) -> JudgeProfile:
    """聊天类应用：要点覆盖（规则）+ 边界行为（规则）+ 可选语气/相关性（LLM）。"""
    dims = ["fact_coverage", "boundary"]
    graders: dict[str, Grader] = {"fact_coverage": rule_fact_coverage, "boundary": rule_boundary}
    if use_llm:
        dims += ["tone", "relevance"]
        graders["tone"] = make_llm_tone(client)
        graders["relevance"] = make_llm_relevance(client)
    return JudgeProfile(profile_id="chat-default", dimensions=dims, graders=graders)


def json_profile() -> JudgeProfile:
    """结构化输出类应用：Schema 合法性 + 字段值核对（全规则、零 LLM 成本）。"""
    return JudgeProfile(
        profile_id="json-default", dimensions=["json_schema_valid", "field_accuracy"],
        graders={"json_schema_valid": rule_json_schema_valid, "field_accuracy": rule_field_accuracy},
    )


def none_profile() -> JudgeProfile:
    """不打分（只跑被测应用、存原始观察）。"""
    return JudgeProfile(profile_id="none", dimensions=[], graders={})


class RagJudgeProfile(JudgeProfile):
    """把现有 RAG 四维评分（evalkit/judge.LLMJudge）包装成通用接口。

    约定：RAG 用例经 evaluation_case_from_rag 转换时，原 Case 字典保存在
    metadata.rag —— 本 Profile 用它还原 Case/TargetResult 后复用 LLMJudge，
    评分语义与旧流水线完全一致。
    """

    def __init__(self, judge=None):
        from evalkit.judge import LLMJudge

        self._llm_judge = judge or LLMJudge()
        super().__init__(profile_id="rag-four-dims",
                         dimensions=["correctness", "faithfulness", "format", "tone"], graders={})

    @property
    def total_usage(self) -> dict:
        return self._llm_judge.total_usage

    def judge(self, case: EvaluationCase, obs: TargetObservation) -> dict:
        from evalkit.schema import Case as RagCase, TargetResult

        rag = case.metadata.get("rag")
        if rag:
            rag_case = RagCase.from_dict(rag)
        else:  # 兜底：直接从通用字段还原
            rag_case = RagCase(id=case.case_id, category=case.category,
                               query=case.input.get("text", ""),
                               key_facts=list(case.expected_behavior.get("facts") or []))
        target = TargetResult(
            answer_text=obs.output, citations=obs.meta.get("citations", {}),
            cited_cards=list(obs.evidence), rejected=obs.status == "rejected",
            citations_ok=bool(obs.meta.get("citations_ok", True)),
            latency_ms=obs.latency_ms, error=obs.error,
        )
        return self._llm_judge.judge(rag_case, target)
