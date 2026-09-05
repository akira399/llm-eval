"""评测平台的数据契约：用例（Case）与被测输出（TargetResult）。

用例集是 YAML 文件（见 cases/*.yaml），load_cases 负责加载与结构校验；
TargetResult 是被测应用适配器的统一输出，Runner 把两者拼成运行记录落盘。
"""
from __future__ import annotations

import yaml
from dataclasses import asdict, dataclass, field

EXPECT_VALUES = ("answer", "reject", "safe")
DIFFICULTY_VALUES = ("easy", "hard")


class CaseError(ValueError):
    """用例结构不合法。"""


@dataclass
class Case:
    id: str
    category: str
    query: str
    expect: str = "answer"
    difficulty: str = "easy"
    key_facts: list[str] = field(default_factory=list)
    notes: str = ""
    expect_card_en: str = ""  # 期望命中的知识卡片（title_en 或 card_id），失败归因用；可省略

    @classmethod
    def from_dict(cls, raw: dict, source: str = "") -> "Case":
        where = f"{source}[{raw.get('id', '?')}]" if source else raw.get("id", "?")
        for key in ("id", "category", "query"):
            if not raw.get(key) or not str(raw[key]).strip():
                raise CaseError(f"{where}: 缺少必填字段 {key}")
        expect = raw.get("expect", "answer")
        if expect not in EXPECT_VALUES:
            raise CaseError(f"{where}: expect 必须是 {EXPECT_VALUES}，得到 {expect!r}")
        difficulty = raw.get("difficulty", "easy")
        if difficulty not in DIFFICULTY_VALUES:
            raise CaseError(f"{where}: difficulty 必须是 {DIFFICULTY_VALUES}，得到 {difficulty!r}")
        key_facts = [str(f) for f in raw.get("key_facts") or [] if str(f).strip()]
        if expect == "answer" and not key_facts:
            raise CaseError(f"{where}: expect=answer 的用例必须有 key_facts（裁判的核对要点）")
        return cls(
            id=str(raw["id"]),
            category=str(raw["category"]),
            query=str(raw["query"]).strip(),
            expect=expect,
            difficulty=difficulty,
            key_facts=key_facts,
            notes=str(raw.get("notes", "")),
            expect_card_en=str(raw.get("expect_card_en", "") or ""),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def load_cases(path: str) -> tuple[dict, list[Case]]:
    """加载用例集，返回 (meta, cases)。结构不合法直接抛错（宁可跑不起来，不跑出脏数据）。"""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise CaseError(f"{path}: 文件必须是 {meta: ..., cases: [...]} 结构")
    meta = data.get("meta") or {}
    cases: list[Case] = []
    seen: set[str] = set()
    for raw in data["cases"]:
        case = Case.from_dict(raw, source=path)
        if case.id in seen:
            raise CaseError(f"{path}: 用例 id 重复：{case.id}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise CaseError(f"{path}: 用例集为空")
    return meta, cases


@dataclass
class TargetResult:
    """被测应用对一次查询的完整输出（适配器的统一格式）。"""

    answer_text: str = ""
    citations: dict[str, str] = field(default_factory=dict)   # {"1": "card_id"} 引用编号→卡片
    cited_cards: list[dict] = field(default_factory=list)     # 被引卡片的原文（忠实度裁判用）
    rejected: bool = False                                    # 是否走了拒答闸
    citations_ok: bool = True                                 # 被测系统自己的引用校验结果
    latency_ms: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
