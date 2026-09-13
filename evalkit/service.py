"""评测服务层：Web/CLI/MCP 共用的唯一业务入口（不让任何入口直连脚本）。

Phase 2.5 新增访问控制地基：
- 租户（tenant）+ API Key（只存 sha256 哈希，明文仅创建时返回一次）；
- 运行产物按租户分目录（runs/tenants/<tenant_id>/），Job 行携带 tenant_id；
- 所有按 id 的读取都校验租户归属——查不到按 NOT_FOUND 返回（不泄露存在性）；
- 本地 MCP/CLI 走默认租户 "local"，不需要 Key（单机可信场景）。

诚实的边界：本阶段仍是 SQLite 单文件、单进程假设；PostgreSQL/RLS/对象存储
按 docs/00 §11.3 路线在 Phase 2 完整版落地。
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import secrets
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
LOCAL_TENANT = "local"


class ServiceError(ValueError):
    """业务级错误：message 面向用户，code 面向程序（MCP/HTTP 映射稳定错误码）。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EvalService:
    def __init__(self, root: str | None = None, db_path: str | None = None):
        self.root = os.path.abspath(root or _ROOT)
        self.cases_dir = os.path.join(self.root, "cases")
        self.suites_dir = os.path.join(self.root, "suites")
        self.runs_dir = os.path.join(self.root, "runs")
        os.makedirs(self.runs_dir, exist_ok=True)
        self.db_path = db_path or os.path.join(self.runs_dir, "jobs.db")
        self._db_lock = threading.Lock()
        self._init_db()
        self._threads: dict[str, threading.Thread] = {}

    # ---------------------------------------------------------------- 存储

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._db_lock, self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                       id TEXT PRIMARY KEY,
                       suite_id TEXT, target_id TEXT, version TEXT,
                       status TEXT, total INTEGER DEFAULT 0, done INTEGER DEFAULT 0,
                       judge INTEGER DEFAULT 1,
                       run_file TEXT, error TEXT,
                       idempotency_key TEXT,
                       created_at TEXT, updated_at TEXT)"""
            )
            # 旧库迁移（已存在则忽略）；必须先于依赖新列的索引创建
            for col, ddl in (
                ("tenant_id", "ALTER TABLE jobs ADD COLUMN tenant_id TEXT DEFAULT 'local'"),
                ("run_id", "ALTER TABLE jobs ADD COLUMN run_id TEXT"),
            ):
                try:
                    conn.execute(ddl)  # 旧库迁移；已存在则忽略
                except sqlite3.OperationalError:
                    pass
            # 幂等键作用域 = 租户（Phase 2.5：跨租户同 key 合法）
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_jobs_tenant_idem "
                         "ON jobs(tenant_id, idempotency_key)")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS tenants (
                       id TEXT PRIMARY KEY, name TEXT, created_at TEXT)"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS api_keys (
                       key_hash TEXT PRIMARY KEY, tenant_id TEXT, name TEXT,
                       revoked INTEGER DEFAULT 0, created_at TEXT)"""
            )

    def _runs_dir_for(self, tenant: str) -> str:
        """local 租户沿用历史目录（兼容既有运行与报表）；其他租户物理隔离。"""
        path = self.runs_dir if tenant == LOCAL_TENANT else \
            os.path.join(self.runs_dir, "tenants", tenant)
        os.makedirs(path, exist_ok=True)
        return path

    # ---------------------------------------------------------------- 租户与 Key

    def create_tenant(self, name: str) -> dict:
        """创建租户并签发首把 API Key。明文只此一次返回，库里只存哈希。"""
        tenant_id = f"tenant_{uuid.uuid4().hex[:8]}"
        api_key = f"llev_{secrets.token_hex(16)}"
        with self._db_lock, self._conn() as conn:
            conn.execute("INSERT INTO tenants (id, name, created_at) VALUES (?,?,?)",
                         (tenant_id, name, _now()))
            conn.execute("INSERT INTO api_keys (key_hash, tenant_id, name, created_at) VALUES (?,?,?,?)",
                         (_sha256(api_key), tenant_id, "default", _now()))
        return {"tenant_id": tenant_id, "api_key": api_key,
                "warning": "api_key 只显示这一次，请立即保存；泄露请用 revoke_api_key 吊销"}

    def authenticate(self, bearer: str | None) -> str:
        """API Key → tenant_id。无效/吊销/缺失一律 AUTH_REQUIRED（不区分原因，防探测）。"""
        if not bearer or not bearer.startswith("llev_"):
            raise ServiceError("AUTH_REQUIRED", "缺少有效的 API Key（Authorization: Bearer llev_…）")
        with self._conn() as conn:
            row = conn.execute("SELECT tenant_id FROM api_keys WHERE key_hash=? AND revoked=0",
                               (_sha256(bearer),)).fetchone()
        if row is None:
            raise ServiceError("AUTH_REQUIRED", "API Key 无效或已吊销")
        return row["tenant_id"]

    def revoke_api_key(self, bearer: str) -> dict:
        with self._db_lock, self._conn() as conn:
            cur = conn.execute("UPDATE api_keys SET revoked=1 WHERE key_hash=?", (_sha256(bearer),))
            if cur.rowcount == 0:
                raise ServiceError("AUTH_REQUIRED", "API Key 无效")
        return {"revoked": True}

    # ---------------------------------------------------------------- 套件仓库

    def _suite_files(self) -> dict[str, tuple[str, str]]:
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

    def list_jobs(self, limit: int = 20, tenant: str = LOCAL_TENANT) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE tenant_id=? ORDER BY created_at DESC LIMIT ?",
                (tenant, max(1, min(limit, 100)))).fetchall()
        return [dict(r) for r in rows]

    def get_job(self, job_id: str, tenant: str = LOCAL_TENANT) -> dict:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=? AND tenant_id=?",
                               (job_id, tenant)).fetchone()
        if row is None:  # 不存在或不属于该租户：统一 NOT_FOUND，不泄露存在性
            raise ServiceError("JOB_NOT_FOUND", f"任务不存在：{job_id}")
        return dict(row)

    def _get_job_row(self, job_id: str) -> dict | None:
        """内部直读（执行线程/更新用，不过滤租户）。"""
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def start_run(self, suite_id: str, target_id: str, version: str, limit: int | None = None,
                  judge: bool = True, budget_max_cases: int | None = None,
                  idempotency_key: str | None = None, tenant: str = LOCAL_TENANT,
                  async_start: bool = True) -> dict:
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
        with self._db_lock, self._conn() as conn:
            if key:
                existed = conn.execute(
                    "SELECT * FROM jobs WHERE idempotency_key=? AND tenant_id=?",
                    (key, tenant)).fetchone()
                if existed:
                    return dict(existed)  # 幂等：同租户内重复提交返回原任务
            conn.execute(
                "INSERT INTO jobs (id, suite_id, target_id, version, status, total, judge,"
                " idempotency_key, tenant_id, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (job_id, suite_id, target_id, version, "queued", len(todo),
                 1 if judge else 0, key, tenant, _now(), _now()),
            )
        if async_start:
            thread = threading.Thread(target=self._execute_job, args=(job_id,), daemon=True)
            self._threads[job_id] = thread
            thread.start()
        return self.get_job(job_id, tenant=tenant)

    # ---------------------------------------------------------------- 任务执行

    def _execute_job(self, job_id: str) -> None:
        """执行任务（公开给测试同步调用；生产由 start_run 的线程调用）。"""
        job = self._get_job_row(job_id)
        if job is None or job["status"] not in ("queued",):
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
                adapter=adapter, profile=profile,
                out_dir=self._runs_dir_for(job.get("tenant_id") or LOCAL_TENANT),
                progress_cb=lambda done, total, cid: self._update(job_id, done=done),
            )
            summary = engine.run(cases, version=job["version"], limit=limit)
            status = "succeeded" if summary["n_errors"] == 0 else "partial"
            self._update(job_id, status=status, run_file=summary["run_file"],
                         run_id=summary.get("run_id"), done=summary["n_cases"])
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
        with self._db_lock, self._conn() as conn:
            conn.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    def cancel_job(self, job_id: str, tenant: str = LOCAL_TENANT) -> dict:
        job = self.get_job(job_id, tenant=tenant)
        if job["status"] in ("queued", "running"):
            # 协作式取消：标记后由引擎在下个检查点停止（Phase 1 引擎尚未接停止位，
            # 此处先将 queued 任务拒之门外、running 任务标记 cancelled 收尾）
            self._update(job_id, status="cancelled", error="用户取消（已完成部分保留在 runs/）")
        return self.get_job(job_id, tenant=tenant)

    # ---------------------------------------------------------------- 运行/对比/归因

    def _find_run(self, version_or_file: str, tenant: str = LOCAL_TENANT) -> str:
        pattern = os.path.join(self._runs_dir_for(tenant), f"{version_or_file}*.jsonl")
        matches = sorted(glob.glob(pattern), key=os.path.getmtime)
        if not matches:
            raise ServiceError("RUN_NOT_FOUND", f"找不到运行记录：{version_or_file}")
        return matches[-1]

    def list_runs(self, tenant: str = LOCAL_TENANT, limit: int = 30) -> list[dict]:
        """列出本租户最近的运行（新→旧），供工作台/报表选择。

        version 取记录内的逻辑版本（meta.version），而非文件名戳——
        用户在工作台看到的是自己起的版本名。
        """
        runs_dir = self._runs_dir_for(tenant)
        out = []
        for path in sorted(glob.glob(os.path.join(runs_dir, "*.jsonl")),
                           key=os.path.getmtime, reverse=True):
            stem = os.path.splitext(os.path.basename(path))[0]
            version = stem
            try:
                with open(path, encoding="utf-8") as f:
                    first = f.readline()
                if first.strip():
                    version = json.loads(first).get("meta", {}).get("version") or stem
            except (OSError, json.JSONDecodeError):
                pass
            out.append({
                "run_file": os.path.basename(path),
                "version": version,
                "mtime": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds"),
            })
            if len(out) >= limit:
                break
        return out

    def get_run(self, version_or_file: str, tenant: str = LOCAL_TENANT) -> dict:
        """运行摘要（不含逐条原文——逐条内容大，按需用 attribute/对比工具取）。"""
        path = self._find_run(version_or_file, tenant=tenant)
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        version = records[0]["meta"]["version"] if records else version_or_file
        summary = summarize(records, version=version)
        summary["run_file"] = os.path.basename(path)
        summary["n_records"] = len(records)
        return summary

    def compare_runs(self, baseline: str, candidate: str, tenant: str = LOCAL_TENANT) -> dict:
        def _agg(path: str) -> dict:
            with open(path, encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
            s = summarize(records, version=os.path.basename(path))
            by_cat = {cat: st.get("correctness_mean") for cat, st in s["by_category"].items()
                      if st.get("correctness_mean") is not None}
            return {"run_file": os.path.basename(path), "n_cases": s["n_cases"],
                    "n_rejected": s["n_rejected"], "n_errors": s["n_errors"],
                    "judge_means": s["judge_means"], "correctness_by_category": by_cat}

        left = _agg(self._find_run(baseline, tenant=tenant))
        right = _agg(self._find_run(candidate, tenant=tenant))
        delta_dims = {dim: round(right["judge_means"].get(dim, 0) - left["judge_means"].get(dim, 0), 3)
                      for dim in sorted(set(left["judge_means"]) | set(right["judge_means"]))}
        return {"baseline": left, "candidate": right, "delta": {"judge_means": delta_dims}}

    def attribute_run(self, version_or_file: str, tenant: str = LOCAL_TENANT) -> dict:
        """两级归因。仅支持含期望卡片的 RAG 套件（generic 套件无卡片语义）。"""
        from evalkit.attribution import attribute_run as _attribute_run

        path = self._find_run(version_or_file, tenant=tenant)
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
        result = _attribute_run(rag_records)
        return {"run_file": os.path.basename(path), "failures": result,
                "n_failures": sum(1 for a in result.values() if a.get("failed"))}


def get_service(root: str | None = None) -> EvalService:
    """服务单例入口（root 可用环境变量 LLM_EVAL_ROOT 覆盖，便于 MCP 子进程指定）。"""
    import os as _os

    return EvalService(root=root or _os.environ.get("LLM_EVAL_ROOT"))
