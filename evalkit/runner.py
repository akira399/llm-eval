"""评测运行器：跑用例集 → 调被测应用 →（可选）四维评分 → 落盘 runs/。

原始结果全部落盘（方案 §2 设计决策 4）：评分可以事后重跑/换裁判模型，
运行不需要重跑。产物：{version}-{时间戳}.jsonl（逐用例）+ .summary.json（汇总）。
"""
from __future__ import annotations

import json
import os
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from evalkit.schema import Case, TargetResult

JUDGE_DIMENSIONS = ("correctness", "faithfulness", "format", "tone")


class Runner:
    def __init__(self, provider, judge=None, out_dir: str | None = None):
        self.provider = provider
        self.judge = judge
        self.out_dir = out_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs"
        )
        os.makedirs(self.out_dir, exist_ok=True)

    def run(self, cases: list[Case], version: str, workers: int = 1, limit: int | None = None) -> dict:
        todo = cases[:limit] if limit else cases
        t0 = datetime.now()
        if workers <= 1:
            records = [self._run_one(case, version) for case in todo]
        else:
            # 被测应用并发能力有限时保持 workers=1（LLM 推理通常 2~4 足够）
            with ThreadPoolExecutor(max_workers=workers) as pool:
                records = list(pool.map(lambda c: self._run_one(c, version), todo))

        stamp = t0.strftime("%Y%m%d-%H%M%S")
        run_path = os.path.join(self.out_dir, f"{version}-{stamp}.jsonl")
        with open(run_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        summary = summarize(records, version=version)
        summary["run_file"] = os.path.basename(run_path)
        summary["effective_config"] = self._effective_config()
        summary_path = run_path.replace(".jsonl", ".summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary

    def _run_one(self, case: Case, version: str) -> dict:
        result: TargetResult = self.provider.answer(case.query)
        judge = None
        judge_error = None
        if self.judge:
            try:
                judge = self.judge.judge(case, result)
            except Exception as exc:  # 评分器异常不拖垮整批（与服务化审查 H3 对齐）
                judge, judge_error = {}, f"{type(exc).__name__}: {exc}"
        record = {
            "meta": {"version": version, "ts": datetime.now().isoformat(timespec="seconds")},
            "case": case.to_dict(),
            "target": result.to_dict(),
            "judge": judge,
        }
        if judge_error:
            record["judge_error"] = judge_error
        return record

    def _effective_config(self) -> dict:
        getter = getattr(self.provider, "effective_config", None)
        try:
            return getter() if getter else {}
        except Exception as exc:
            return {"error": str(exc)}


def summarize(records: list[dict], version: str) -> dict:
    """一组运行记录的统计：总数/作答/拒答/错误、耗时、四维均分、分类目拆分。"""
    n = len(records)
    answered = [r for r in records if not r["target"]["rejected"] and not r["target"]["error"]]
    rejected = sum(1 for r in records if r["target"]["rejected"])
    errors = sum(1 for r in records if r["target"]["error"])
    latencies = [r["target"]["latency_ms"] for r in records if not r["target"]["error"]]

    summary: dict = {
        "version": version,
        "n_cases": n,
        "n_answered": len(answered),
        "n_rejected": rejected,
        "n_errors": errors,
        "latency_ms_avg": round(statistics.mean(latencies)) if latencies else None,
        "by_difficulty": {},
        "by_category": {},
        "judge_means": {},
    }

    for difficulty in ("easy", "hard"):
        subset = [r for r in records if r["case"]["difficulty"] == difficulty]
        if subset:
            summary["by_difficulty"][difficulty] = _stats(subset)

    categories = sorted({r["case"]["category"] for r in records})
    for cat in categories:
        summary["by_category"][cat] = _stats([r for r in records if r["case"]["category"] == cat])

    # 维度动态收集：旧 RAG 四维与通用 JudgeProfile 的任意维度统一处理
    dims_present = sorted({d for r in records if r.get("judge") for d in r["judge"]})
    for dim in dims_present:
        scores = [
            r["judge"][dim]["score"]
            for r in records
            if r.get("judge") and r["judge"].get(dim) and r["judge"][dim].get("score") is not None
        ]
        if scores:
            summary["judge_means"][dim] = round(statistics.mean(scores), 3)
    return summary


def _stats(records: list[dict]) -> dict:
    """一组记录的统计：数量、作答/拒答/错误、各维度均分。"""
    out: dict = {"n": len(records)}
    out["n_rejected"] = sum(1 for r in records if r["target"]["rejected"])
    out["n_errors"] = sum(1 for r in records if r["target"]["error"])
    for dim in sorted({d for r in records if r.get("judge") for d in r["judge"]}):
        scores = [
            r["judge"][dim]["score"]
            for r in records
            if r.get("judge") and r["judge"].get(dim) and r["judge"][dim].get("score") is not None
        ]
        if scores:
            out[f"{dim}_mean"] = round(statistics.mean(scores), 3)
    return out
