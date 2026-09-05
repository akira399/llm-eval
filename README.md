# LLM-Eval · LLM 应用效果评测与回归平台

给 LLM 应用装上「效果仪表盘」：改提示词/换模型/调检索参数之前，
先用同一套用例集跑一遍，用数字回答三个问题——**变好了还是变坏了？
问题出在检索还是生成？AI 裁判自己可信吗？**

首个被测对象是同仓库的 [Poke-RAG](../poke-rag)（宝可梦对战知识库问答），
适配器模式让它可以评测任何可编程调用的 LLM 应用。

## 架构

```
用例集（50 条种子，10 类目标签）
   → Runner（记录原始输出 runs/*.jsonl，评分可离线重跑）
   → 被测适配器（进程内驱动 RAGEngine / HTTP SSE，版本=配置快照）
   → Judge 四维评分（正确性/引用忠实度=LLM 裁判；格式/拒答行为=确定性规则）
   → 汇总对比（分类目统计、版本回归、失败归因）
```

## 快速开始

复用 poke-rag 的虚拟环境（被测系统同解释器）：

```bash
# Git Bash
PY=../poke-rag/.venv/Scripts/python.exe

# 冒烟：3 条，不做 LLM 裁判
$PY scripts/run_eval.py --limit 3 --no-judge

# 全量 50 条 + 四维裁判（DeepSeek，成本几分钱）
$PY scripts/run_eval.py --version bm25-baseline

# 版本回归：换配置跑第二版，然后对比
$PY scripts/run_eval.py --override '{"rag": {"top_k_answer": 8}}' --version topk8
$PY scripts/summarize.py "runs/bm25-baseline-*.jsonl" "runs/topk8-*.jsonl"

# promptfoo 链路（阶段 0 的交叉验证）
$PY scripts/export_promptfoo_tests.py
npx promptfoo eval -c promptfoo/promptfooconfig.yaml && npx promptfoo view

# 人工盲评 → 一致率 → 报告（"AI 裁判可信"的证据链）
$PY -m streamlit run scripts/annotate_app.py      # 逐条盲评（界面隐藏 AI 分数）
$PY scripts/agreement.py                          # 一致率 + Cohen's Kappa
$PY -m streamlit run scripts/show_report.py --server.port=8502
$PY scripts/make_report.py                        # 产出 reports/评测报告-*.md

# 单元测试
$PY -m pytest tests/ -q
```

## 用例集格式

```yaml
- id: dex-001
  category: 图鉴
  query: 快龙是什么属性的宝可梦？
  expect: answer        # answer / reject（应承认不知道）/ safe（注入类，不得越界）
  difficulty: easy      # hard=已知弱项，单独统计
  key_facts:            # 裁判逐条核对的参考要点
    - 指出快龙是龙属性和飞行属性
```

## 文档

[技术方案](docs/00-技术方案.md)（四维指标定义、Judge 一致性协议、失败归因、里程碑）

## 路线图

- **M0（已完成）**：平台骨架 + 50 条种子用例 + 适配器/Runner/Judge + promptfoo 接线
- **M1（工具就绪，待人工标注）**：盲评标注界面 + 一致率/Kappa 计算 + Streamlit 报表 + 报告生成器
- **M2**：版本回归真实数据 + 失败归因报告
- **M3**：种子扩写 30→200 + 反馈回流格式 + 红队用例 + 评测报告 PDF
