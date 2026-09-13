"""通用评测引擎（Phase 0）：TargetAdapter + JudgeProfile → 与旧 Runner 同形的记录与汇总。

与 evalkit/runner.py 的关系：Runner 是 RAG 流水线的回归基线（保留不动）；
Engine 是泛化后的执行路径——任何实现 TargetAdapter 的应用、任何 JudgeProfile
都能跑，产物（jsonl/summary）与旧 Runner 完全同形，报表/对比/归因工具直接复用。

可靠性约定（与 Runner 一致并对齐服务化审查结论）：
- 单条用例的适配器异常/评分器异常都被隔离为该条记录的 error/score=None，
  绝不中断整批；
- 每条完成即回调 progress_cb（Phase 1 的 Job 检查点用）；
- 产物文件名仍为 {version}-{秒级时间戳}，Phase 1 引入 run_id 后再消歧。
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from time import perf_counter

from evalkit.contracts import EvaluationCase, InvocationRequest, TargetAdapter, TargetObservation
from evalkit.judge_profile import JudgeProfile


class EvaluationEngine:
    def __init__(self, adapter: TargetAdapter, profile: JudgeProfile,
                 out_dir: str | None = None, progress_cb=None):
        self.adapter = adapter
        self.profile = profile
        self.out_dir = out_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs"
        )
        os.makedirs(self.out_dir, exist_ok=True)
        self.progress_cb = progress_cb  # (done, total, case_id)

    def run(self, cases: list[EvaluationCase], version: str,
            workers: int = 1, limit: int | None = None) -> dict:
        from evalkit.runner import summarize  # 复用同一套汇总（动态维度）

        todo = cases[:limit] if limit else cases
        t0 = datetime.now()
        if workers <= 1:
            records = [self._run_one(case, version, i, len(todo)) for i, case in enumerate(todo, 1)]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                records = list(pool.map(
                    lambda pair: self._run_one(pair[1], version, pair[0], len(todo)),
                    enumerate(todo, 1),
                ))

        stamp = t0.strftime("%Y%m%d-%H%M%S")
        run_path = os.path.join(self.out_dir, f"{version}-{stamp}.jsonl")
        with open(run_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        summary = summarize(records, version=version)
        summary["run_file"] = os.path.basename(run_path)
        summary["engine"] = {"target_id": self.adapter.target_id, "profile_id": self.profile.profile_id}
        summary_path = run_path.replace(".jsonl", ".summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary

    def _run_one(self, case: EvaluationCase, version: str, index: int, total: int) -> dict:
        request = InvocationRequest(
            input=dict(case.input),
            metadata={"case_id": case.case_id, "version": version},
        )
        t0 = perf_counter()
        try:
            obs = self.adapter.invoke(request)
            if not isinstance(obs, TargetObservation):
                raise TypeError(f"适配器返回了 {type(obs).__name__}，期望 TargetObservation")
        except Exception as exc:  # 适配器异常 = 该条用例失败，不拖垮整批
            obs = TargetObservation(status="error", error=f"{type(exc).__name__}: {exc}")
        if not obs.latency_ms:
            obs.latency_ms = int((perf_counter() - t0) * 1000)

        judge = self.profile.judge(case, obs)
        if self.progress_cb:
            try:
                self.progress_cb(index, total, case.case_id)
            except Exception:
                pass  # 进度回调失败不影响评测本身

        return {
            "meta": {"version": version, "ts": datetime.now().isoformat(timespec="seconds")},
            "case": case.to_dict(),
            "target": {**obs.to_dict(), **obs.legacy_fields()},
            "judge": judge,
        }
