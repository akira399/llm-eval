"""服务层测试：套件清单、Job 状态机、幂等、预算硬限制、运行/对比/归因。"""
import json
import os
import shutil

import pytest

from evalkit.service import EvalService, ServiceError

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def service(tmp_path):
    """临时 root：拷贝演示套件，隔离 runs 与 jobs.db。"""
    os.makedirs(os.path.join(str(tmp_path), "suites"), exist_ok=True)
    for name in ("demo-chat.yaml", "demo-json.yaml"):
        shutil.copy(os.path.join(_REPO, "suites", name),
                    os.path.join(str(tmp_path), "suites", name))
    return EvalService(root=str(tmp_path))


def test_list_suites_scans_cases_and_suites(service):
    suites = {s["suite_id"]: s for s in service.list_suites()}
    assert suites["demo-chat"]["case_count"] == 6
    assert suites["demo-json"]["case_count"] == 5
    assert suites["demo-chat"]["kind"] == "generic"
    assert len(suites["demo-chat"]["sha256"]) == 64


def test_get_suite_pagination(service):
    suite = service.get_suite("demo-chat", limit=2)
    assert suite["case_count"] == 6 and suite["cases_returned"] == 2
    with pytest.raises(ServiceError) as ei:
        service.get_suite("nope")
    assert ei.value.code == "SUITE_NOT_FOUND"


def test_start_run_job_lifecycle(service):
    job = service.start_run(suite_id="demo-chat", target_id="demo-chat",
                            version="svc-chat-v1", judge=True)
    assert job["status"] in ("queued", "running")
    service._execute_job(job["id"])  # 测试同步执行（生产由线程跑）
    done = service.get_job(job["id"])
    assert done["status"] == "succeeded" and done["done"] == 6
    run = service.get_run("svc-chat-v1")
    assert run["n_cases"] == 6
    assert 0 < run["judge_means"]["fact_coverage"] <= 1.0


def test_start_run_without_judge(service):
    job = service.start_run(suite_id="demo-chat", target_id="demo-chat",
                            version="svc-nojudge", judge=False)
    service._execute_job(job["id"])
    run = service.get_run("svc-nojudge")
    assert run["judge_means"] == {}  # 不打分：只存原始观察


def test_budget_hard_limit(service):
    with pytest.raises(ServiceError) as ei:
        service.start_run(suite_id="demo-chat", target_id="demo-chat",
                          version="over-budget", budget_max_cases=3)
    assert ei.value.code == "BUDGET_EXCEEDED"


def test_unknown_suite_and_target(service):
    with pytest.raises(ServiceError) as e1:
        service.start_run(suite_id="nope", target_id="demo-chat", version="x")
    assert e1.value.code == "SUITE_NOT_FOUND"
    with pytest.raises(ServiceError) as e2:
        service.start_run(suite_id="demo-chat", target_id="http://evil.example", version="x")
    assert e2.value.code == "TARGET_NOT_REGISTERED"  # 任意 URL/目标被拒绝


def test_idempotency_key_returns_same_job(service):
    a = service.start_run(suite_id="demo-chat", target_id="demo-chat",
                          version="idem", idempotency_key="key-001")
    b = service.start_run(suite_id="demo-chat", target_id="demo-chat",
                          version="idem", idempotency_key="key-001")
    assert a["id"] == b["id"]


def test_compare_runs_delta(service):
    for version in ("cmp-a", "cmp-b"):
        job = service.start_run(suite_id="demo-chat", target_id="demo-chat", version=version)
        service._execute_job(job["id"])
    result = service.compare_runs("cmp-a", "cmp-b")
    assert result["baseline"]["n_cases"] == result["candidate"]["n_cases"] == 6
    assert result["delta"]["judge_means"]["fact_coverage"] == 0.0  # 确定性应用两次一致


def test_attribute_run_rejects_generic_suite(service):
    job = service.start_run(suite_id="demo-chat", target_id="demo-chat", version="attr-x")
    service._execute_job(job["id"])
    with pytest.raises(ServiceError) as ei:
        service.attribute_run("attr-x")
    assert ei.value.code == "UNSUPPORTED"


def test_attribute_run_on_rag_records(service):
    """手工构造 RAG 形状的运行记录：归因工具应能按期望卡片定位失败。"""
    run_file = os.path.join(service.runs_dir, "fake-rag-run-20260913.jsonl")
    os.makedirs(os.path.dirname(run_file), exist_ok=True)
    rag_case = {"id": "meta-001", "category": "对战环境", "query": "Gen9 OU 热门是谁？",
                "expect": "answer", "difficulty": "easy", "key_facts": ["具体宝可梦名"],
                "notes": "", "expect_card_en": "meta:gen9ou"}
    record = {"meta": {"version": "fake-rag-run", "ts": "t"}, "case": rag_case,
              "target": {"answer_text": "知识库中未找到相关信息。", "citations": {},
                         "cited_cards": [], "rejected": False, "citations_ok": True,
                         "latency_ms": 5, "error": ""},
              "judge": {"correctness": {"score": 0.0}, "faithfulness": {"score": None},
                        "format": {"score": 0.0}, "tone": {"score": None}}}
    with open(run_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    result = service.attribute_run("fake-rag-run")
    assert result["n_failures"] >= 1
    assert "meta-001" in result["failures"]
