"""反馈回流测试：格式校验、去重、增强挂点、候选渲染。"""
import json

from evalkit.feedback import candidates_to_yaml_lines, ingest, normalize_query, parse_feedback
from evalkit.schema import Case


def _write(tmp_path, lines: list[str]) -> str:
    path = str(tmp_path / "fb.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def test_parse_feedback_validates(tmp_path):
    path = _write(tmp_path, [
        json.dumps({"feedback_id": "fb-1", "ts": "t", "query": "q1", "rating": "down"}),
        json.dumps({"feedback_id": "fb-2", "ts": "t", "query": "q2", "rating": "meh"}),
        json.dumps({"feedback_id": "fb-3", "rating": "down"}),  # 缺 ts/query
        "not json",
        "",
    ])
    records, errors = parse_feedback(path)
    assert len(records) == 1 and records[0]["feedback_id"] == "fb-1"
    assert len(errors) == 3


def test_ingest_dedup_and_up_skip():
    existing = [Case(id="c1", category="图鉴", query="十万伏特的威力是多少？", key_facts=["90"])]
    records = [
        {"feedback_id": "fb-1", "ts": "t", "query": "十万伏特的威力", "rating": "up"},
        {"feedback_id": "fb-2", "ts": "t", "query": "十万伏特的威力是多少", "rating": "down"},  # 归一化后与已有重复
        {"feedback_id": "fb-3", "ts": "t", "query": "耿鬼速度多少？", "rating": "down"},
        {"feedback_id": "fb-4", "ts": "t", "query": "耿鬼速度多少", "rating": "down"},  # 候选内去重
    ]
    candidates, stats = ingest(records, existing)
    assert stats["up_skipped"] == 1
    assert stats["duplicate"] == 2
    assert len(candidates) == 1
    assert candidates[0]["source_feedback"] == "fb-3"
    assert "待构建要点" in candidates[0]["status"]


def test_ingest_with_enricher():
    records = [{"feedback_id": "fb-9", "ts": "t", "query": "新问题？", "rating": "down"}]
    def enricher(query):
        return {"card_id": "poke:149", "facts": ["要点一", "要点二"], "support_ratio": 0.9}
    candidates, stats = ingest(records, [], enricher=enricher)
    assert stats["enriched"] == 1
    assert candidates[0]["card_id"] == "poke:149"


def test_yaml_lines_render_facts_and_notes():
    candidates = [{"query": "q?", "source_feedback": "fb-1", "comment": "", "app_version": "v",
                   "status": "已按卡片锚定生成要点（机器预筛，仍需人工复核）",
                   "card_id": "poke:149", "facts": ["要点一"]}]
    lines = candidates_to_yaml_lines(candidates)
    text = "\n".join(lines)
    assert "fb-candidate-001" in text and "要点一" in text and "poke:149" in text


def test_normalize_query():
    assert normalize_query("十万伏特，威力？") == normalize_query("十万伏特威力")
