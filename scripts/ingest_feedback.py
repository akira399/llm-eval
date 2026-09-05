"""反馈回流 CLI（M3）：点踩数据 → 用例集候选。

用法：
  # 基础回流：去重 + 生成候选草稿
  ../poke-rag/.venv/Scripts/python.exe scripts/ingest_feedback.py examples/feedback-sample.jsonl
  # 带要点增强：检索 top-1 卡片做事实锚点 + LLM 生成参考要点（机器预筛）
  ../poke-rag/.venv/Scripts/python.exe scripts/ingest_feedback.py examples/feedback-sample.jsonl --enrich

产物：cases/feedback-candidates.yaml（人工逐条复核后并入正式用例集）
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalkit.feedback import candidates_to_yaml_lines, ingest, parse_feedback  # noqa: E402
from evalkit.schema import load_cases  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_enricher():
    """检索 top-1 卡片 + LLM 从卡片原文生成要点（卡片锚定，与扩写同纪律）。"""
    from evalkit.expander import SUPPORT_PASS, case_support
    from evalkit.judge import _parse_json

    sys.path.insert(0, os.path.abspath(os.path.join(_ROOT, "..", "poke-rag")))
    from src.retrieval.index import search  # noqa: E402
    from src.generation.prompt import load_cards

    cards = load_cards()

    def enricher(query: str) -> dict | None:
        hits = search(query, top_k=1)
        if not hits:
            return None
        card = cards.get(hits[0][0])
        if not card or len(card.get("content_zh", "")) < 40:
            return None
        from evalkit import llm as llm_mod

        raw = llm_mod.chat([
            {"role": "system", "content": (
                "你是评测集构建专家。用户对这个回答点了踩，请基于知识卡片原文"
                "构建该查询的评测要点。要点必须能从卡片原文找到依据，严禁编造。只输出 JSON。")},
            {"role": "user", "content": (
                f"用户查询：{query}\n检索命中的卡片 [{card['card_id']}] {card['title_zh']}：\n"
                f"{card['content_zh'][:600]}\n\n"
                '输出 JSON：{"facts": [<1~3条参考要点>]}')},
        ], temperature=0.2)
        try:
            facts = [str(f).strip() for f in _parse_json(raw).get("facts") or [] if str(f).strip()]
        except (ValueError, KeyError, TypeError):
            return None
        if not facts:
            return None
        probe = type("C", (), {"key_facts": facts})  # 复用 case_support 的鸭子类型
        ratio = case_support(probe, card["content_zh"])
        if ratio < SUPPORT_PASS:
            return None
        return {"card_id": card["card_id"], "facts": facts, "support_ratio": round(ratio, 2)}

    return enricher


def main() -> int:
    parser = argparse.ArgumentParser(description="点踩反馈回流")
    parser.add_argument("feedback_file", help="反馈 JSONL 文件")
    parser.add_argument("--enrich", action="store_true", help="检索锚点 + LLM 生成要点")
    args = parser.parse_args()

    records, errors = parse_feedback(args.feedback_file)
    for err in errors:
        print(f"⚠️ {err}")
    all_cases: list = []
    for name in sorted(os.listdir(os.path.join(_ROOT, "cases"))):
        if name.endswith(".yaml") and not name.startswith("feedback"):
            _, cases = load_cases(os.path.join(_ROOT, "cases", name))
            all_cases.extend(cases)

    enricher = make_enricher() if args.enrich else None
    candidates, stats = ingest(records, all_cases, enricher=enricher)

    out_path = os.path.join(_ROOT, "cases", "feedback-candidates.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(candidates_to_yaml_lines(candidates)) + "\n")

    print(f"\n===== 反馈回流 =====")
    print(f"反馈 {stats['total']} 条 · 点踩候选 {len(candidates)} 条 · "
          f"点赞跳过 {stats['up_skipped']} · 与已有用例重复 {stats['duplicate']}")
    if args.enrich:
        print(f"要点增强：成功 {stats['enriched']} · 未命中可用卡片 {stats['pending']}")
    print(f"候选草稿：{out_path}（逐条人工复核后并入正式用例集）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
