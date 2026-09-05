"""人机一致性计算（方案 §5）：人工盲评 vs AI 裁判。

协议：人工按与 AI 不同的尺度打分（好理解的自然语言档位），本模块把
AI 的连续分数分箱到同一档位后，逐维度计算一致率与 Cohen's Kappa，
并列出分歧清单——分歧样本是修订评分细则的原料。

分箱规则（人工档位 ↔ AI 分数）：
  correctness   对 / 部分对 / 错          ↔ ≥0.75 / 0.25~0.75 / <0.25
  faithfulness  没有编造 / 有编造嫌疑 / 明显编造 ↔ ≥0.9 / 0.5~0.9 / <0.5
  format        没问题 / 有问题           ↔ ≥0.75 / <0.75
  tone          好 / 一般 / 差            ↔ 2 / 1 / 0
拒答类（AI 忠实度=skip）不参与忠实度比对。
"""
from __future__ import annotations

import json
import os
from collections import Counter

from evalkit.runner import JUDGE_DIMENSIONS

HUMAN_LABELS = {
    "correctness": ("对", "部分对", "错"),
    "faithfulness": ("没有编造", "有编造嫌疑", "明显编造"),
    "format": ("没问题", "有问题"),
    "tone": ("好", "一般", "差"),
}


def bin_ai(dimension: str, score) -> str | None:
    """AI 连续分数 → 人工档位。None（skip/出错）返回 None，不参与比对。"""
    if score is None:
        return None
    if dimension == "correctness":
        return "对" if score >= 0.75 else ("部分对" if score >= 0.25 else "错")
    if dimension == "faithfulness":
        return "没有编造" if score >= 0.9 else ("有编造嫌疑" if score >= 0.5 else "明显编造")
    if dimension == "format":
        return "没问题" if score >= 0.75 else "有问题"
    if dimension == "tone":
        return {2: "好", 1: "一般", 0: "差"}.get(int(score), "一般")
    return None


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float | None:
    """Cohen's Kappa。类别完全一致时（pe=1）按定义返回 1.0。"""
    if not labels_a or len(labels_a) != len(labels_b):
        return None
    n = len(labels_a)
    po = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    ca, cb = Counter(labels_a), Counter(labels_b)
    categories = set(ca) | set(cb)
    pe = sum(ca.get(c, 0) * cb.get(c, 0) for c in categories) / (n * n)
    if pe >= 1.0:
        return 1.0
    return round((po - pe) / (1 - pe), 3)


def annotations_path(run_file: str, annotations_dir: str | None = None) -> str:
    stem = os.path.splitext(os.path.basename(run_file))[0]
    base = annotations_dir or os.path.join(os.path.dirname(run_file), "annotations")
    return os.path.join(base, f"{stem}.jsonl")


def load_annotations(path: str) -> dict[str, dict]:
    """case_id → 人工标注。同 case 多条时取最后一条（可改判）。"""
    if not os.path.exists(path):
        return {}
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["case_id"]] = rec
    return out


def save_annotation(path: str, record: dict) -> None:
    """整文件重写（条数少，简单可靠）：同一条用例以最后一次保存为准。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing = load_annotations(path)
    existing[record["case_id"]] = record
    with open(path, "w", encoding="utf-8") as f:
        for rec in existing.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def compute_agreement(records: list[dict], annotations: dict[str, dict]) -> dict:
    """逐维度：可比对数、一致率、Kappa、分歧清单。"""
    by_case = {r["case"]["id"]: r for r in records}
    result: dict = {"n_annotated": len(annotations), "dimensions": {}}
    for dim in JUDGE_DIMENSIONS:
        human_labels: list[str] = []
        ai_labels: list[str] = []
        disagreements: list[dict] = []
        for case_id, ann in annotations.items():
            record = by_case.get(case_id)
            if record is None:
                continue
            human = (ann.get("human") or {}).get(dim)
            ai_score = ((record.get("judge") or {}).get(dim) or {}).get("score")
            ai = bin_ai(dim, ai_score)
            if human is None or ai is None:
                continue
            human_labels.append(human)
            ai_labels.append(ai)
            if human != ai:
                disagreements.append({
                    "case_id": case_id,
                    "human": human,
                    "ai": ai,
                    "ai_score": ai_score,
                    "notes": ann.get("notes", ""),
                })
        n = len(human_labels)
        dim_out: dict = {"n_compared": n}
        if n:
            agree = sum(h == a for h, a in zip(human_labels, ai_labels))
            dim_out["agreement"] = round(agree / n, 3)
            dim_out["kappa"] = cohen_kappa(human_labels, ai_labels)
            dim_out["disagreements"] = disagreements
        result["dimensions"][dim] = dim_out
    return result
