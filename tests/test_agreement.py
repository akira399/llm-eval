"""一致性模块测试：Kappa 数学、分箱规则、标注读写、端到端一致率计算。"""
import json

from evalkit.agreement import (
    annotations_path,
    bin_ai,
    cohen_kappa,
    compute_agreement,
    load_annotations,
    save_annotation,
)
from evalkit.schema import Case, TargetResult


def test_cohen_kappa_known_values():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0      # 完全一致
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "a", "b", "b"]) == 0.0      # 交叉对齐 → 0
    # 手算验证：po=3/4，pe=(2*3+2*1)/16=1/2，kappa=(0.75-0.5)/0.5=0.5
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "a", "b", "a"]) == 0.5
    # 一方判分恒定时，超出随机的一致为零（Kappa 定义如此）
    assert cohen_kappa(["a", "a", "a"], ["a", "a", "b"]) == 0.0
    assert cohen_kappa([], []) is None


def test_bin_ai_boundaries():
    assert bin_ai("correctness", 1.0) == "对"
    assert bin_ai("correctness", 0.75) == "对"
    assert bin_ai("correctness", 0.5) == "部分对"
    assert bin_ai("correctness", 0.2) == "错"
    assert bin_ai("faithfulness", 0.95) == "没有编造"
    assert bin_ai("faithfulness", 0.6) == "有编造嫌疑"
    assert bin_ai("faithfulness", 0.3) == "明显编造"
    assert bin_ai("format", 0.8) == "没问题"
    assert bin_ai("format", 0.5) == "有问题"
    assert bin_ai("tone", 2) == "好"
    assert bin_ai("tone", 1) == "一般"
    assert bin_ai("tone", 0) == "差"
    assert bin_ai("correctness", None) is None  # skip 不参与比对


def test_annotation_roundtrip_and_last_wins(tmp_path):
    path = str(tmp_path / "ann.jsonl")
    save_annotation(path, {"case_id": "a-1", "human": {"tone": "好"}, "notes": ""})
    save_annotation(path, {"case_id": "b-1", "human": {"tone": "差"}, "notes": ""})
    save_annotation(path, {"case_id": "a-1", "human": {"tone": "一般"}, "notes": "改判"})
    annotations = load_annotations(path)
    assert annotations["a-1"]["human"]["tone"] == "一般"   # 同一条以最后一次为准
    assert annotations["b-1"]["human"]["tone"] == "差"


def _record(case_id: str, judge: dict) -> dict:
    case = Case(id=case_id, category="图鉴", query="q", key_facts=["f"])
    return {"meta": {"version": "v"}, "case": case.to_dict(),
            "target": TargetResult().to_dict(), "judge": judge}


def test_compute_agreement_end_to_end():
    records = [
        _record("c1", {"correctness": {"score": 1.0}, "faithfulness": {"score": 1.0},
                       "format": {"score": 1.0}, "tone": {"score": 2}}),
        _record("c2", {"correctness": {"score": 0.0}, "faithfulness": {"score": None},
                       "format": {"score": 0.5}, "tone": {"score": 0}}),
    ]
    annotations = {
        "c1": {"case_id": "c1", "human": {"correctness": "对", "faithfulness": "没有编造",
                                          "format": "没问题", "tone": "好"}, "notes": ""},
        "c2": {"case_id": "c2", "human": {"correctness": "对", "faithfulness": "没有编造",
                                          "format": "没问题", "tone": "好"}, "notes": "分歧"},
    }
    result = compute_agreement(records, annotations)
    assert result["n_annotated"] == 2
    dims = result["dimensions"]
    assert dims["correctness"]["n_compared"] == 2
    assert dims["correctness"]["agreement"] == 0.5
    assert dims["correctness"]["disagreements"][0]["case_id"] == "c2"
    assert dims["faithfulness"]["n_compared"] == 1          # c2 忠实度 skip，不比对
    assert dims["tone"]["disagreements"][0]["human"] == "好"
    assert dims["tone"]["disagreements"][0]["ai"] == "差"


def test_annotations_path_layout(tmp_path):
    run_file = str(tmp_path / "runs" / "v1-20260906-120000.jsonl")
    expected = str(tmp_path / "runs" / "annotations" / "v1-20260906-120000.jsonl")
    assert annotations_path(run_file).replace("\\", "/") == expected.replace("\\", "/")


def test_annotation_file_stays_valid_jsonl(tmp_path):
    path = str(tmp_path / "ann.jsonl")
    save_annotation(path, {"case_id": "x", "human": {}, "notes": "含\"引号\"和\n换行"})
    with open(path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 1 and lines[0]["notes"] == "含\"引号\"和\n换行"
