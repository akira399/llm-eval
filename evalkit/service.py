"""评测服务层（Phase 1）：Web/CLI/MCP 共用的唯一业务入口（方案：不让 MCP 直连脚本）。

职责：
- 套件仓库：扫描 cases/*.yaml（RAG）与 suites/*.yaml（generic），内容哈希寻址；
- 目标注册表：只接受已注册 target_id（不收任意路径/URL，服务化安全边界）；
- Job 模型：SQLite 持久化任务（状态机 + 进度 + 幂等键），后台线程执行；
- 运行/对比/归因：复用 Runner 与 EvaluationEngine 的产物，工具层零业务逻辑。

安全约定（对齐 docs/00 §11 服务化审查）：
- 所有产物读写经本服务按 id 定位，绝不把客户端路径拼进文件系统；
- effective_config 由适配器脱敏后才入 summary；
- Job 属于创建它的服务实例（本地单用户阶段无租户，Phase 3 引入 tenant_id）。
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime

from evalkit.contracts import evaluation_case_from_rag, load_evaluation_cases
from evalkit.engine import EvaluationEngine
from evalkit.judge_profile import RagJudgeProfile, chat_profile, json_profile, none_profile
from evalkit.registry import TARGET_CATALOG, build_target
from evalkit.runner import summarize
from evalkit.schema import load_cases

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JOB_STATES = ("queued", "running", "cancelling", "cancelled", "succeeded", "partial", "failed")


class ServiceError(ValueError):
    """业务级错误：message 面向用户，code 面向程序（MCP 工具映射稳定错误码）。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class EvalService:
    def __init__(self, root: str | None = None, db_path: str | None = None):
        self.root = os.path.abspath(root or _ROOT)
        self.cases_dir = os.path.join(self.root, "cases")
        self.suites_dir = os.path.join(self.root, "suites")
        self.runs_dir = os.path.join(self.root, "runs")
        os.makedirs(self.runs_dir, exist_ok=True)
        self.db_path = db_path or os.path.join(self.runs_dir, "jobs.db")
        self._init_db()
        self._threads: dict[str, threading.Thread] = {}

    # ---------------------------------------------------------------- 套件仓库

    def _suite_files(self) -> dict[str, tuple[str, str]]:
        """suite_id → (path, kind)。cases/=RAG 专用套件，suites/=generic 套件。"""
        found: dict[str, tuple[str, str]] = {}
        for folder, kind in ((self.cases_dir, "rag"), (self.suites_dir, "generic")):
            if not os.path.isdir(folder):
                continue
            for name in sorted(os.listdir(folder)):
                if name.endswith(".yaml") and not name.startswith("feedback"):
                    found.setdefault(name[:-5], (os.path.join(folder, name), kind))
        return found

    def list_suites(self) -> list[dict]:
        out = []
        for suite_id, (path, kind) in self._suite_files().items():
            sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
            try:
                _, cases = self._load_suite(path, kind)
                categories: dict[str, int] = {}
                for c in cases:
                    categories[c.category] = categories.get(c.category, 0) + 1
                info = {"suite_id": suite_id, "kind": kind, "sha256": sha,
                        "case_count": len(cases), "categories": categories}
            except Exception as exc:
                info = {"suite_id": suite_id, "kind": kind, "sha256": sha,
                        "error": f"套件加载失败：{exc}"}
            out.append(info)
        return sorted(out, key=lambda s: s["suite_id"])

    def _load_suite(self, path: str, kind: str):
        """按套件类型加载为统一 EvaluationCase 列表（RAG 套件经契约转换）。"""
        if kind == "generic":
            return load_evaluation_cases(path)
        _, rag_cases = load_cases(path)
        return None, [evaluation_case_from_rag(c) for c in rag_cases]

    def get_suite(self, suite_id: str, limit: int = 20) -> dict:
        files = self._suite_files()
        if suite_id not in files:
            raise ServiceError("SUITE_NOT_FOUND", f"评测集不存在：{suite_id}")
        path, kind = files[suite_id]
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
        meta, cases = self._load_suite(path, kind)
        items = [c.to_dict() for c in cases[:max(0, min(limit, 100))]]
        return {"suite_id": suite_id, "kind": kind, "sha256": sha, "meta": meta,
                "case_count": len(cases), "cases_returned": len(items), "cases": items}

    # ---------------------------------------------------------------- Job 模型

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                       id TEXT PRIMARY KEY,
                       suite_id TEXT, target_id TEXT, version TEXT,
                       status TEXT, total INTEGER DEFAULT 0, done INTEGER DEFAULT 0,
                       judge INTEGER DEFAULT 1,
                       run_file TEXT, error TEXT,
                       idempotency_key TEXT UNIQUE,
                       created_at TEXT, updated_at TEXT)"""
            )

    def list_jobs(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 100)),)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_job(self, job_id: str) -> dict:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise ServiceError("JOB_NOT_FOUND", f"任务不存在：{job_id}")
        return dict(row)

    def start_run(self, suite_id: str, target_id: str, version: str, limit: int | None = None,
                  judge: bool = True, budget_max_cases: int | None = None,
                  idempotency_key: str | None = None, async_start: bool = True) -> dict:
        """创建评测任务并立即返回（长任务异步化：绝不阻塞调用方跑完 183 条）。

        async_start=False 仅供测试同步执行（不启动后台线程）。
        """
        files = self._suite_files()
        if suite_id not in files:
            raise ServiceError("SUITE_NOT_FOUND", f"评测集不存在：{suite_id}")
        if target_id not in TARGET_CATALOG:
            raise ServiceError("TARGET_NOT_REGISTERED",
                               f"未注册的被测目标：{target_id}（可选：{sorted(TARGET_CATALOG)}）")
        _, kind = files[suite_id]
        _, cases = self._load_suite(*files[suite_id])
        todo = cases[:limit] if limit else cases
        if budget_max_cases is not None and len(todo) > budget_max_cases:
            raise ServiceError("BUDGET_EXCEEDED",
                               f"本次将运行 {len(todo)} 条，超过预算 {budget_max_cases}；"
                               "请调大 budget_max_cases 或用 limit 缩小范围")

        job_id = f"job_{uuid.uuid4().hex[:12]}"
        key = idempotency_key or None
        with self._conn() as conn:
            if key:
                existed = conn.execute("SELECT * FROM jobs WHERE idempotency_key=?", (key,)).fetchone()
                if existed:
                    return dict(existed)  # 幂等：重复提交返回原任务
            conn.execute(
                "INSERT INTO jobs (id, suite_id, target_id, version, status, total, judge,"
                " idempotency_key, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (job_id, suite_id, target_id, version, "queued", len(todo),
                 1 if judge else 0, key, _now(), _now()),
            )
        thread = None
        if async_start:
            thread = threading.Thread(target=self._execute_job, args=(job_id,), daemon=True)
            self._threads[job_id] = thread
            thread.start()
        return self.get_job(job_id)

    # ---------------------------------------------------------------- 任务执行

    def _execute_job(self, job_id: str) -> None:
        """执行任务（公开给测试同步调用；生产由 start_run 的线程调用）。"""
        job = self.get_job(job_id)
        if job["status"] not in ("queued",):
            return
        self._update(job_id, status="running")
        try:
            files = self._suite_files()
            path, kind = files[job["suite_id"]]
            _, cases = self._load_suite(path, kind)
            limit = None
            if job["total"]:
                cases = cases[: job["total"]]
                limit = job["total"]

            adapter = build_target(job["target_id"])
            # 评分配置按"目标应用的类型"选（chat/json/rag），与套件文件类型解耦
            target_kind = TARGET_CATALOG.get(job["target_id"], {}).get("kind", "chat")
            profile = self._pick_profile(target_kind, bool(job["judge"]))
            engine = EvaluationEngine(
                adapter=adapter, profile=profile, out_dir=self.runs_dir,
                progress_cb=lambda done, total, cid: self._update(job_id, done=done),
            )
            summary = engine.run(cases, version=job["version"], limit=limit)
            status = "succeeded" if summary["n_errors"] == 0 else "partial"
            self._update(job_id, status=status, run_file=summary["run_file"], done=summary["n_cases"])
        except Exception as exc:
            self._update(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")

    def _pick_profile(self, kind: str, judge_enabled: bool):
        if not judge_enabled:
            return none_profile()
        return {"chat": chat_profile, "json": json_profile, "rag": RagJudgeProfile}.get(
            kind, none_profile)()

    def _update(self, job_id: str, **fields) -> None:
        fields["updated_at"] = _now()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._conn() as conn:
            conn.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    def cancel_job(self, job_id: str) -> dict:
        job = self.get_job(job_id)
        if job["status"] in ("queued", "running"):
            # 协作式取消：标记后由引擎在下个检查点停止（Phase 1 引擎尚未接停止位，
            # 此处先将 queued 任务拒之门外、running 任务标记 cancelled 收尾）
            self._update(job_id, status="cancelled", error="用户取消（已完成部分保留在 runs/）")
        return self.get_job(job_id)

    # ---------------------------------------------------------------- 运行/对比/归因

    def _find_run(self, version_or_file: str) -> str:
        pattern = os.path.join(self.runs_dir, f"{version_or_file}*.jsonl")
        matches = sorted(glob.glob(pattern), key=os.path.getmtime)
        if not matches:
            raise ServiceError("RUN_NOT_FOUND", f"找不到运行记录：{version_or_file}")
        return matches[-1]

    def get_run(self, version_or_file: str) -> dict:
        """运行摘要（不含逐条原文——逐条内容大，按需用 attribute/对比工具取）。"""
        path = self._find_run(version_or_file)
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        version = records[0]["meta"]["version"] if records else version_or_file
        summary = summarize(records, version=version)
        summary["run_file"] = os.path.basename(path)
        summary["n_records"] = len(records)
        return summary

    def compare_runs(self, baseline: str, candidate: str) -> dict:
        def _agg(path: str) -> dict:
            with open(path, encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
            s = summarize(records, version=os.path.basename(path))
            by_cat = {cat: st.get("correctness_mean") for cat, st in s["by_category"].items()
                      if st.get("correctness_mean") is not None}
            return {"run_file": os.path.basename(path), "n_cases": s["n_cases"],
                    "n_rejected": s["n_rejected"], "n_errors": s["n_errors"],
                    "judge_means": s["judge_means"], "correctness_by_category": by_cat}

        left, right = _agg(self._find_run(baseline)), _agg(self._find_run(candidate))
        delta_dims = {dim: round(right["judge_means"].get(dim, 0) - left["judge_means"].get(dim, 0), 3)
                      for dim in sorted(set(left["judge_means"]) | set(right["judge_means"]))}
        return {"baseline": left, "candidate": right, "delta": {"judge_means": delta_dims}}

    def attribute_run(self, version_or_file: str) -> dict:
        """两级归因。仅支持含期望卡片的 RAG 套件（generic 套件无卡片语义）。"""
        from evalkit.attribution import attribute_run as _attribute_run
        from evalkit.schema import Case as RagCase

        path = self._find_run(version_or_file)
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        rag_records = []
        for r in records:
            case_dict = r.get("case") or {}
            rag_meta = (case_dict.get("metadata") or {}).get("rag")  # 引擎记录：包装在 metadata.rag
            if rag_meta:
                rag_records.append({**r, "case": rag_meta})
            elif "id" in case_dict and "expect" in case_dict:  # 旧 Runner 记录：本身就是 RAG 用例
                rag_records.append(r)
            else:
                raise ServiceError("UNSUPPORTED", "归因目前仅支持 RAG 套件（含 expect_card_en）的运行记录")
        # 记录里已内嵌当时的 RAG 用例快照，直接归因（无需外部套件）
        result = _attribute_run(rag_records)
        return {"run_file": os.path.basename(path), "failures": result,
                "n_failures": sum(1 for a in result.values() if a.get("failed"))}


def get_service(root: str | None = None) -> EvalService:
    """服务单例入口（root 可用环境变量 LLM_EVAL_ROOT 覆盖，便于 MCP 子进程指定）。"""
    import os as _os

    return EvalService(root=root or _os.environ.get("LLM_EVAL_ROOT"))
