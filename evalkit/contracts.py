"""通用评测契约（Phase 0）：与具体 AI 应用解耦的输入/输出/用例模型。

设计原则（docs/00-技术方案.md §11 泛化路线）：
- InvocationRequest / TargetObservation 是"连接器协议"：任何 AI 应用实现
  TargetAdapter.invoke 即可接入评测，Engine/Judge 不感知对方内部细节；
- EvaluationCase 的 input / expected_behavior 是通用 JSON；RAG 场景的
  query / key_facts / citations 属于能力扩展（capability），不是所有应用必备；
- 旧 Case / TargetResult（RAG 专用）通过 convert_* 与新契约互通，现有 RAG
  流水线保持不变，作为回归基线。

套件文件格式（generic）：
    meta: {name: ..., format: generic, ...}
    cases:
      - case_id: chat-001
        input: {text: "..."}            # 或 {messages: [...]} 等
        expected_behavior:
          type: answer|reject|safe|json|any
          facts: ["要点"]               # answer 类可选，规则评分器用
          json_schema: {...}            # json 类必填
          expect_fields: {k: v}         # json 类可选，字段值精确核对
        category: 售后
        difficulty: easy
        tags: []
        metadata: {}
"""
from __future__ import annotations

import yaml
from dataclasses import asdict, dataclass, field

KNOWN_BEHAVIOR_TYPES = ("answer", "reject", "safe", "json", "any")
KNOWN_DIFFICULTY = ("easy", "hard")


class ContractError(ValueError):
    """通用契约数据不合法。"""


# ---------------------------------------------------------------- 请求与观察

@dataclass
class InvocationRequest:
    """发给被测应用的一次调用请求（通用信封）。"""

    input: dict                                   # {"text": ...} / {"messages": [...]} 等
    variables: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    timeout_s: int = 120                          # 连接器应尽力遵守；本仓库演示应用为即时返回

    def text(self) -> str:
        """便捷取文本输入（text 或 messages 最后一条 user 内容）。"""
        if "text" in self.input:
            return str(self.input["text"])
        messages = self.input.get("messages") or []
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))
        return ""


@dataclass
class TargetObservation:
    """被测应用输出的统一观察结果（通用信封）。

    status: success | rejected | error
    evidence: 通用证据列表（RAG 应用 = 被引用的知识卡片）
    meta: 附加信息（如 citations 映射、trace_id）
    """

    status: str = "success"
    output: str = ""
    structured_output: dict | None = None
    evidence: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    latency_ms: int = 0
    error: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def legacy_fields(self) -> dict:
        """兼容 runner.summarize / 旧报表的 RAG 形状字段。"""
        return {
            "answer_text": self.output,
            "citations": self.meta.get("citations", {}),
            "cited_cards": list(self.evidence),
            "rejected": self.status == "rejected",
            "citations_ok": bool(self.meta.get("citations_ok", True)),
        }


# ---------------------------------------------------------------- 适配器协议

class TargetAdapter:
    """被测应用连接器协议：实现 invoke 即可接入评测引擎。"""

    target_id: str = "unknown"

    def invoke(self, request: InvocationRequest) -> TargetObservation:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------- 通用用例

@dataclass
class EvaluationCase:
    """通用评测用例：input/expected_behavior 是 JSON，能力扩展放 expected_behavior。"""

    case_id: str
    input: dict
    expected_behavior: dict = field(default_factory=dict)
    category: str = "default"
    difficulty: str = "easy"
    tags: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict, source: str = "") -> "EvaluationCase":
        where = f"{source}[{raw.get('case_id', '?')}]" if source else raw.get("case_id", "?")
        if not raw.get("case_id") or not str(raw["case_id"]).strip():
            raise ContractError(f"{where}: 缺少 case_id")
        if not isinstance(raw.get("input"), dict) or not raw["input"]:
            raise ContractError(f"{where}: input 必须是非空对象（如 {{text: ...}}）")
        behavior = raw.get("expected_behavior") or {}
        if not isinstance(behavior, dict):
            raise ContractError(f"{where}: expected_behavior 必须是对象")
        btype = behavior.get("type", "any")
        if btype not in KNOWN_BEHAVIOR_TYPES:
            raise ContractError(f"{where}: expected_behavior.type 必须是 {KNOWN_BEHAVIOR_TYPES}，得到 {btype!r}")
        if btype == "json" and not isinstance(behavior.get("json_schema"), dict):
            raise ContractError(f"{where}: type=json 的用例必须提供 json_schema")
        difficulty = raw.get("difficulty", "easy")
        if difficulty not in KNOWN_DIFFICULTY:
            raise ContractError(f"{where}: difficulty 必须是 {KNOWN_DIFFICULTY}，得到 {difficulty!r}")
        return cls(
            case_id=str(raw["case_id"]),
            input=raw["input"],
            expected_behavior=behavior,
            category=str(raw.get("category", "default")),
            difficulty=difficulty,
            tags=[str(t) for t in raw.get("tags") or []],
            metadata=raw.get("metadata") or {},
        )

    def to_dict(self) -> dict:
        return asdict(self)


def load_evaluation_cases(path: str) -> tuple[dict, list[EvaluationCase]]:
    """加载 generic 格式套件，返回 (meta, cases)。结构与 RAG 的 load_cases 同风格。"""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ContractError(f"{path}: 文件必须是 {{meta: ..., cases: [...]}} 结构")
    meta = data.get("meta") or {}
    cases: list[EvaluationCase] = []
    seen: set[str] = set()
    for raw in data["cases"]:
        case = EvaluationCase.from_dict(raw, source=path)
        if case.case_id in seen:
            raise ContractError(f"{path}: case_id 重复：{case.case_id}")
        seen.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ContractError(f"{path}: 用例集为空")
    return meta, cases


# ---------------------------------------------------------------- 新旧契约互通

def evaluation_case_from_rag(case) -> "EvaluationCase":
    """旧 RAG Case → 通用 EvaluationCase（原字段存 metadata.rag 供 RAG 评分器还原）。"""
    return EvaluationCase(
        case_id=case.id,
        input={"text": case.query},
        expected_behavior={"type": case.expect, "facts": list(case.key_facts)},
        category=case.category,
        difficulty=case.difficulty,
        tags=["rag"],
        metadata={"rag": case.to_dict(), "expect_card_en": getattr(case, "expect_card_en", "")},
    )


def observation_from_target_result(result) -> "TargetObservation":
    """旧 TargetResult → 通用 TargetObservation。"""
    status = "error" if result.error else ("rejected" if result.rejected else "success")
    return TargetObservation(
        status=status,
        output=result.answer_text,
        evidence=list(result.cited_cards),
        latency_ms=result.latency_ms,
        error=result.error,
        meta={"citations": dict(result.citations), "citations_ok": result.citations_ok},
    )
