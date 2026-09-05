"""Runner 测试：用假被测对象验证运行/落盘/汇总逻辑（不碰真实系统与 LLM）。"""
import json
import os

from evalkit.runner import Runner
from evalkit.schema import Case, TargetResult


class FakeProvider:
    """按 query 前缀 canned 三种行为：正常带引用 / 拒答 / 出错。"""

    def __init__(self):
        self.calls: list[str] = []

    def answer(self, query: str) -> TargetResult:
        self.calls.append(query)
        if "错误" in query:
            return TargetResult(error="RuntimeError: boom")
        if "天气" in query:
            return TargetResult(answer_text="知识库中未找到相关信息。", rejected=True, latency_ms=5)
        return TargetResult(
            answer_text=f"关于「{query}」的回答 [1]。",
            citations={"1": "card-1"},
            cited_cards=[{"card_id": "card-1", "title_zh": "卡一", "content_zh": "内容一"}],
            citations_ok=True,
            latency_ms=10,
        )

    def effective_config(self):
        return {"fake": True}


def _cases() -> list[Case]:
    return [
        Case(id="a-1", category="图鉴", query="快龙是什么属性？", key_facts=["龙属性"]),
        Case(id="b-1", category="超范围", query="今天天气怎么样？", expect="reject"),
        Case(id="c-1", category="图鉴", query="触发错误"),
    ]


def test_run_writes_records_and_summary(tmp_path):
    provider = FakeProvider()
    runner = Runner(provider=provider, out_dir=str(tmp_path))
    summary = runner.run(_cases(), version="fake-v1")

    assert provider.calls == ["快龙是什么属性？", "今天天气怎么样？", "触发错误"]
    assert summary["n_cases"] == 3
    assert summary["n_answered"] == 1
    assert summary["n_rejected"] == 1
    assert summary["n_errors"] == 1
    assert summary["effective_config"] == {"fake": True}

    run_path = os.path.join(str(tmp_path), summary["run_file"])
    with open(run_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 3
    assert records[0]["case"]["id"] == "a-1"
    assert records[0]["target"]["citations"] == {"1": "card-1"}
    assert os.path.exists(run_path.replace(".jsonl", ".summary.json"))


def test_summary_per_category_and_difficulty(tmp_path):
    runner = Runner(provider=FakeProvider(), out_dir=str(tmp_path))
    cases = _cases()
    cases[0].difficulty = "hard"
    summary = runner.run(cases, version="fake-v2")
    assert summary["by_category"]["图鉴"]["n"] == 2
    assert summary["by_difficulty"]["hard"]["n"] == 1
    # 无 judge 时四维均分不出现
    assert summary["judge_means"] == {}


def test_with_judge_scores_recorded(tmp_path):
    class FakeJudge:
        def judge(self, case, result):
            return {
                "correctness": {"score": 1.0 if not result.rejected else None},
                "faithfulness": {"score": None},
                "format": {"score": 1.0},
                "tone": {"score": 2},
            }

    runner = Runner(provider=FakeProvider(), judge=FakeJudge(), out_dir=str(tmp_path))
    summary = runner.run(_cases()[:1], version="fake-v3")
    assert summary["judge_means"]["correctness"] == 1.0
    assert summary["judge_means"]["tone"] == 2.0
    assert "faithfulness" not in summary["judge_means"]  # 全 None 的维度不进均分


def test_limit_runs_subset(tmp_path):
    provider = FakeProvider()
    runner = Runner(provider=provider, out_dir=str(tmp_path))
    summary = runner.run(_cases(), version="fake-v4", limit=2)
    assert summary["n_cases"] == 2
    assert len(provider.calls) == 2
