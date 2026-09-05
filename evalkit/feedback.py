"""线上反馈回流（M3）：把真实用户的点踩数据沉淀进用例集。

数据格式（JSONL，每行一条）：
  {"feedback_id": "fb-001", "ts": "2026-09-07T10:00:00", "query": "...",
   "rating": "down", "comment": "回答说错了", "app_version": "poke-rag@v3"}

回流纪律（与扩写同源）：
- 只收 rating=down 的查询作为候选（点踩 = 线上评测的"失败用例"）；
- 与现有用例集按归一化 query 去重；
- 候选条目必须经过「要点构建 + 人工复核」才能进入正式用例集——
  enrich 模式用检索 top-1 卡片做事实锚点生成参考要点，同样带机器支持率预筛。
"""
from __future__ import annotations

import json
import re

from evalkit.schema import Case

REQUIRED_FIELDS = ("feedback_id", "ts", "query", "rating")
VALID_RATINGS = ("up", "down")


def normalize_query(q: str) -> str:
    return re.sub(r"[\s，。？！?!,.、：:\"'（）()]", "", q or "").lower()


def parse_feedback(path: str) -> tuple[list[dict], list[str]]:
    """读取并校验反馈文件，返回 (合法记录, 错误列表)。"""
    records, errors = [], []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"第{i}行不是合法 JSON：{exc}")
                continue
            missing = [k for k in REQUIRED_FIELDS if not rec.get(k)]
            if missing:
                errors.append(f"第{i}行缺少字段：{missing}")
                continue
            if rec["rating"] not in VALID_RATINGS:
                errors.append(f"第{i}行 rating 必须是 {VALID_RATINGS}，得到 {rec['rating']!r}")
                continue
            records.append(rec)
    return records, errors


def ingest(records: list[dict], existing_cases: list[Case],
           enricher=None) -> tuple[list[dict], dict]:
    """筛出值得进用例集的候选。

    enricher: 可选的 (query) -> {"card_id":..., "content_zh":..., "facts":[...]}，
    通常由「检索 top-1 卡片 + LLM 从卡片原文生成要点」实现（scripts 层注入）。
    返回 (候选列表, 统计)。
    """
    known = {normalize_query(c.query) for c in existing_cases}
    candidates: list[dict] = []
    seen_queries: set[str] = set()
    stats = {"total": len(records), "up_skipped": 0, "duplicate": 0, "enriched": 0, "pending": 0}
    for rec in records:
        if rec["rating"] != "down":
            stats["up_skipped"] += 1
            continue
        q = normalize_query(rec["query"])
        if q in known or q in seen_queries:
            stats["duplicate"] += 1
            continue
        seen_queries.add(q)
        candidate: dict = {
            "query": rec["query"].strip(),
            "source_feedback": rec["feedback_id"],
            "comment": rec.get("comment", ""),
            "app_version": rec.get("app_version", ""),
            "status": "待构建要点（人工复核后方可入正式用例集）",
        }
        if enricher is not None:
            enriched = enricher(rec["query"].strip())
            if enriched:
                candidate.update(enriched)
                candidate["status"] = "已按卡片锚定生成要点（机器预筛，仍需人工复核）"
                stats["enriched"] += 1
            else:
                stats["pending"] += 1
        else:
            stats["pending"] += 1
        candidates.append(candidate)
    return candidates, stats


def candidates_to_yaml_lines(candidates: list[dict]) -> list[str]:
    """候选写成人可读的 YAML 草稿（人工复核后并入正式用例集）。"""
    lines = ["# 反馈回流候选（机器生成草稿，逐条人工复核后再并入正式用例集）", "cases:"]
    for i, c in enumerate(candidates, 1):
        lines.append(f"  - id: fb-candidate-{i:03d}")
        lines.append(f"    query: {json.dumps(c['query'], ensure_ascii=False)}")
        lines.append("    expect: answer")
        lines.append("    difficulty: easy")
        if c.get("facts"):
            lines.append("    key_facts:")
            for fact in c["facts"]:
                lines.append(f"      - {fact}")
        else:
            lines.append("    key_facts: []  # 待构建")
        if c.get("card_id"):
            lines.append(f"    expect_card_en: {c['card_id']}")
        lines.append(f"    notes: 来源反馈 {c['source_feedback']} · {c['status']}")
    return lines
