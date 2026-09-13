# LLM-Eval · LLM 应用效果评测与回归平台

给 LLM 应用装上「效果仪表盘」：改提示词/换模型/调检索参数之前，
先用同一套用例集跑一遍，用数字回答三个问题——**变好了还是变坏了？
问题出在检索还是生成？AI 裁判自己可信吗？**

- 🧪 **用例集**：183 条（11 类目，卡片锚定扩写）+ 20 条红队用例（注入/越狱/越权/误导）
- ⚖️ **四维评分**：正确性 / 引用忠实度（LLM 裁判）+ 格式 / 拒答行为（确定性规则，零成本）
- 🎯 **裁判可信性**：人工盲评校准协议，人机一致率 98%，Cohen's Kappa 0.935
- 🔍 **失败归因**：检索层 vs 生成层两级自动定位；**版本回归**：同一用例集跑 v1/v2/v3 出对比数字
- 📄 **完整评测报告**：[reports/LLM应用效果评测报告.md](reports/LLM应用效果评测报告.md)（含全部数字与复现命令）

首个被测对象是 [Poke-RAG](https://github.com/akira399/poke-rag)（宝可梦对战知识库问答系统）：
评测发现的 3 处知识缺口驱动其修复，正确率 0.82 → 0.93。适配器模式让它可以
评测任何可编程调用的 LLM 应用。

## 架构

```
用例集（yaml：查询 + 参考要点 + 期望行为 + 期望命中卡片）
   → Runner（记录原始输出 runs/*.jsonl，评分可离线重跑）
   → 被测适配器（进程内驱动 / HTTP SSE；「版本」= 被测系统配置快照）
   → Judge 四维评分
   → 汇总对比（分类目统计 · 版本回归 · 失败归因 · 人工盲评一致率）
```

## 快速开始

本框架依赖一个被测对象。以 Poke-RAG 为例（Python 3.12+）：

```bash
# 1. 获取被测系统并准备运行环境（详见其 README）
git clone https://github.com/akira399/poke-rag
cd poke-rag && pip install -r requirements.txt
python scripts/build_cards.py && python scripts/build_index.py   # 构建知识库
python -m venv .venv && source .venv/Scripts/activate            # Windows Git Bash

# 2. 获取本框架（放在 poke-rag 同级目录，或用 POKE_RAG_ROOT 指定路径）
git clone https://github.com/akira399/llm-eval
cd ../llm-eval
pip install -r requirements.txt

# 3. 配置 LLM（裁判与被测系统共用，OpenAI 兼容协议）
#    poke-rag/config.local.json 或 llm-eval/config.local.json 填 base_url / api_key / model

PY=../poke-rag/.venv/Scripts/python.exe

# 冒烟：3 条，不做 LLM 裁判
$PY scripts/run_eval.py --limit 3 --no-judge

# 全量 183 条 + 四维裁判
$PY scripts/run_eval.py --cases cases/poke-rag-v1.yaml --version my-v1

# 版本回归：改配置跑第二版，然后对比
$PY scripts/run_eval.py --override '{"rag": {"top_k_answer": 8}}' --version topk8
$PY scripts/summarize.py "runs/my-v1-*.jsonl" "runs/topk8-*.jsonl"

# 红队 20 条
$PY scripts/run_eval.py --cases cases/poke-rag-redteam.yaml --version redteam

# 人工盲评 → 一致率 → 报告（"AI 裁判可信"的证据链）
$PY -m streamlit run scripts/annotate_app.py      # 逐条盲评（界面隐藏 AI 分数）
$PY scripts/agreement.py                          # 一致率 + Cohen's Kappa
$PY scripts/attribute.py runs/my-v1-*.jsonl       # 失败归因
$PY scripts/make_report.py                        # 产出 reports/评测报告-*.md

# 单元测试
$PY -m pytest tests/ -q
```

环境变量 `POKE_RAG_ROOT` 可指向任意位置的 poke-rag；适配器 `evalkit/providers/`
是开放的——为其他 LLM 应用写一个同类适配器即可复用全部评测能力。

## 通用评测（任意 AI 应用）

Phase 0 起核心契约与具体应用解耦（`evalkit/contracts.py` + `engine.py` +
`judge_profile.py`）：实现一个 `TargetAdapter.invoke()` 就能评测任何 LLM 应用，
评分维度由 `JudgeProfile` 配置（规则评分器零成本，LLM 评分器可选）。
仓库自带两个离线确定性演示：

```bash
$PY scripts/run_generic_eval.py --list-targets
$PY scripts/run_generic_eval.py --suite suites/demo-chat.yaml --target demo-chat --version demo1
$PY scripts/run_generic_eval.py --suite suites/demo-json.yaml --target demo-json --version demo2
```

## MCP 服务（本地 stdio）

把评测能力暴露给 Claude/Cursor 等 MCP 客户端（9 个工具：列套件/查详情/发起任务/
轮询进度/查结果/版本对比/失败归因等；后台 Job 模型，长任务异步执行）：

```json
{"mcpServers": {"llm-eval": {
  "command": "<python>",
  "args": ["<仓库>/scripts/mcp_server.py"],
  "env": {"LLM_EVAL_ROOT": "<仓库路径>"}
}}}
```

客户端里即可对话式操作："列出评测集 → 发起一次评测 → 查进度 → 解读结果"。
后续路线（FastAPI 多租户服务 / 远程 MCP / 连接器生态）见 [docs/00 §11](docs/00-技术方案.md)。

## 用例集格式

```yaml
- id: dex-001
  category: 图鉴
  query: 快龙是什么属性的宝可梦？
  expect: answer        # answer / reject（应承认不知道）/ safe（注入类，不得越界）
  difficulty: easy      # hard=已知弱项，单独统计
  expect_card_en: dragonite   # 期望命中的知识卡片（失败自动归因用）
  key_facts:            # 裁判逐条核对的参考要点
    - 指出快龙是龙属性和飞行属性
```

扩写器（`scripts/expand_cases.py`）以知识卡片原文为事实锚点生成新用例：
要点优先照抄原文关键词，经分词重叠率预筛——评测集自身的事实可靠性
不依赖 LLM 的诚实度。

## 文档

[技术方案](docs/00-技术方案.md)（四维指标定义、Judge 一致性协议、失败归因设计）
· [最终评测报告](reports/LLM应用效果评测报告.md)
· [版本回归对比 v1→v3](reports/回归对比-v1-v3-20260906.md)
· [扩写人工抽查清单示例](reports/扩写抽查清单.md)

## 成果速览

| 成果 | 数字 |
| --- | --- |
| 用例集 | 183 条（11 类目）+ 20 条红队用例 |
| AI 裁判可信性 | 人机一致率 98%，Kappa 0.935 |
| 评测驱动改进 | 被测应用正确率 0.82→0.93（三轮，每轮有归因证据） |
| 引用忠实度 | 1.0（全程零编造）；红队 20/20 防住 |

## 许可

[MIT](LICENSE) · 数据来源与致谢见 [Poke-RAG](https://github.com/akira399/poke-rag)
