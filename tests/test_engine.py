"""通用引擎端到端测试：演示应用全离线跑通，产物与旧 Runner 同形。"""
import json
import os

from evalkit.contracts import EvaluationCase, InvocationRequest, load_evaluation_cases
from evalkit.demo import DemoChatAdapter, DemoJsonAdapter
from evalkit.engine import EvaluationEngine
from evalkit.judge_profile import chat_profile, json_profile, none_profile
from evalkit.registry import build_target

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name: str):
    return load_evaluation_cases(os.path.join(_ROOT, "suites", name))


def test_chat_suite_end_to_end(tmp_path):
    _, cases = _load("demo-chat.yaml")
    engine = EvaluationEngine(adapter=DemoChatAdapter(), profile=chat_profile(), out_dir=str(tmp_path))
    summary = engine.run(cases, version="demo-chat-test")

    assert summary["n_cases"] == 6
    assert summary["n_errors"] == 0
    assert summary["judge_means"]["fact_coverage"] == 1.0   # FAQ 全命中
    assert summary["judge_means"]["boundary"] == 1.0        # 越界全拒答
    assert summary["engine"]["target_id"] == "demo-chat"


def test_json_suite_end_to_end(tmp_path):
    _, cases = _load("demo-json.yaml")
    engine = EvaluationEngine(adapter=DemoJsonAdapter(), profile=json_profile(), out_dir=str(tmp_path))
    summary = engine.run(cases, version="demo-json-test")

    assert summary["n_cases"] == 5
    # 评分器必须抓出预埋缺陷：json-002/005 缺字段（schema）、json-003 缺 name（字段核对）
    assert 0 < summary["judge_means"]["json_schema_valid"] < 1
    assert 0 < summary["judge_means"]["field_accuracy"] < 1
    # 逐条核对缺陷被定位
    with open(os.path.join(str(tmp_path), summary["run_file"]), encoding="utf-8") as f:
        records = {json.loads(line)["case"]["case_id"]: json.loads(line) for line in f if line.strip()}
    assert records["json-001"]["judge"]["json_schema_valid"]["score"] == 1.0
    assert records["json-002"]["judge"]["json_schema_valid"]["score"] == 0.0
    assert records["json-003"]["judge"]["field_accuracy"]["score"] == 0.5  # 缺 name，age 正确
    assert records["json-005"]["judge"]["field_accuracy"]["score"] is None  # 未声明期望字段 → skip


def test_record_shape_compatible_with_legacy_tools(tmp_path):
    """产物必须与旧 Runner 同形：报表/对比/归因工具无需改动即可消费。"""
    _, cases = _load("demo-chat.yaml")
    engine = EvaluationEngine(adapter=DemoChatAdapter(), profile=chat_profile(), out_dir=str(tmp_path))
    summary = engine.run(cases[:2], version="shape-test")
    with open(os.path.join(str(tmp_path), summary["run_file"]), encoding="utf-8") as f:
        rec = json.loads(f.readline())
    assert set(rec) == {"meta", "case", "target", "judge"}
    assert rec["target"]["answer_text"]          # legacy 别名
    assert rec["target"]["rejected"] is False
    assert rec["case"]["category"] and rec["case"]["difficulty"]


def test_adapter_exception_isolated(tmp_path):
    class ExplodingAdapter:
        target_id = "boom"

        def invoke(self, request):
            if "炸" in request.text():
                raise RuntimeError("被测应用崩溃")
            from evalkit.contracts import TargetObservation

            return TargetObservation(status="success", output="ok")

    from evalkit.judge_profile import JudgeProfile

    cases = [EvaluationCase(case_id=f"c{i}", input={"text": t})
             for i, t in enumerate(["正常", "炸一下", "再正常"], 1)]
    engine = EvaluationEngine(adapter=ExplodingAdapter(), profile=none_profile(), out_dir=str(tmp_path))
    summary = engine.run(cases, version="boom-test")
    assert summary["n_errors"] == 1 and summary["n_cases"] == 3  # 单条失败不拖垮整批


def test_progress_callback_and_limit(tmp_path):
    _, cases = _load("demo-chat.yaml")
    seen: list[tuple] = []
    engine = EvaluationEngine(adapter=DemoChatAdapter(), profile=none_profile(),
                              out_dir=str(tmp_path), progress_cb=lambda d, t, cid: seen.append((d, t)))
    summary = engine.run(cases, version="progress-test", limit=3)
    assert summary["n_cases"] == 3
    assert [d for d, _ in seen] == [1, 2, 3] and all(t == 3 for _, t in seen)


def test_registry_target_runs_through_engine(tmp_path):
    """注册表 → 适配器 → 引擎 全链路（用 demo 目标，不依赖本地环境）。"""
    adapter = build_target("demo-chat")
    _, cases = _load("demo-chat.yaml")
    engine = EvaluationEngine(adapter=adapter, profile=chat_profile(), out_dir=str(tmp_path))
    summary = engine.run(cases[:1], version="registry-test")
    assert summary["judge_means"]["fact_coverage"] == 1.0
