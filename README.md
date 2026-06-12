# COT 思维链推理实验项目

> 基于大语言模型的思维链推理（Chain-of-Thought Reasoning）进阶探索，借鉴 Harness Engineering 设计思想系统化实现与管理 CoT 策略。

---

## 📋 项目简介

本项目基于 OpenAI-compatible API（DMXAPI / deepseek-v4-flash），在 AQuA（Algebraic Word Problems）数据集上实现并对比 6 种 CoT 推理策略：

1. **Base COT** — 基础思维链推理
2. **Self-Consistency** — 多路径采样 + 多数投票
3. **Prefix Consistency** — 截断再生可靠性加权投票
4. **Step-Aware Verifier** — 步骤级验证与最优路径筛选
5. **RAG + COT** — 检索增强思维链
6. **Multi-Agent Debate** — 多 Agent 协作辩论推理

**核心设计**：借鉴 [Harness Engineering](https://github.com/walkinglabs/learn-harness-engineering) 的五子系统思想（Instructions / Tools / Environment / State / Feedback），将每种 CoT 策略映射到 Harness 子系统，系统化地设计、实现与评估推理框架。

---

## 🏗️ 项目结构

```
.
├── data/
│   ├── AQuA/                   # 🔗 git submodule → deepmind/AQuA
│   └── knowledge_base.json     # RAG 检索知识库
├── prompts/                    # CoT 策略 Prompt 模板
│   ├── base_cot.txt
│   ├── rag_cot.txt
│   └── step_verifier.txt
├── strategies/                 # 6 种 CoT 策略实现
│   ├── base.py                 # 策略基类（含 Harness 子系统声明）
│   ├── base_cot.py
│   ├── self_consistency.py
│   ├── prefix_consistency.py   # 截断再生可靠性加权投票
│   ├── step_verifier.py
│   ├── rag_cot.py
│   └── multi_agent_debate.py
├── tasks/                      # 任务环境定义
│   ├── aqua_task.py            # AQuA 数据集加载与评估
│   └── base.py
├── models/                     # LLM 接口封装
│   ├── base.py
│   └── openai_api.py           # OpenAI-compatible API 封装
├── eval/                       # 评估指标与分析工具
│   ├── metrics.py              # 准确率、推理步数、Token 估算
│   └── analyze.py              # 多实验对比分析
├── experiments/                # 实验记录与报告
│   ├── runs/                   # 原始运行结果（JSON）
│   ├── report_small_sample_20260610.md
│   └── report_50_sample_20260611.md
├── retrieval/                  # 检索模块
│   └── simple_retriever.py     # 基于关键词的 TF-IDF 检索
├── scripts/                    # 运行脚本
│   ├── init.sh                 # 环境验证脚本
│   ├── run_all_50.sh           # 50 样本批量测试脚本（Bash）
│   └── run_experiments.py      # 50 样本批量测试脚本（Python）
├── tests/                      # 功能验证测试
│   ├── verify_feat004.py
│   ├── verify_feat005.py
│   ├── verify_feat006.py
│   ├── verify_feat007.py
│   ├── verify_feat008.py
│   ├── verify_feat009.py
│   └── verify_feat010.py
├── docs/                       # 项目文档
│   ├── CLAUDE.md               # 开发规范与指令
│   ├── progress.md             # 开发进度日志
│   ├── session-handoff.md      # 多会话交接记录
│   └── 选题说明.md
├── harness.py                  # 🚀 实验管理主入口
├── harness_report.py           # Harness 子系统覆盖矩阵报告
├── requirements.txt
└── feature_list.json           # 功能状态追踪
```

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
# 带 submodule 克隆（推荐）
git clone --recurse-submodules git@github.com:SHenpengYU01/nlp-cot.git
cd nlp-cot

# 如果已克隆但忘了 --recurse-submodules
git submodule update --init --recursive
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

依赖：`openai>=1.0.0`、`tqdm`、`numpy`

### 3. 环境验证

```bash
bash scripts/init.sh
```

验证内容包括：Python 环境、目录结构、核心文件完整性、模块导入、各策略干跑测试、功能状态列表。

### 4. 配置 API Key

```bash
# Linux / macOS / Git Bash
export DMX_API_KEY="your-api-key-here"

# Windows CMD
set DMX_API_KEY=your-api-key-here

# Windows PowerShell
$env:DMX_API_KEY="your-api-key-here"
```

---

## 🧪 运行实验

### 单策略快速测试

```bash
# 基础 COT（默认 5 条样本）
python harness.py --strategy base_cot --dataset aqua

# RAG + COT（检索 top-3 知识）
python harness.py --strategy rag_cot --dataset aqua --top_k 3

# Self-Consistency（3 条推理路径 + 投票）
python harness.py --strategy self_consistency --dataset aqua --n_paths 3

# Prefix Consistency（3 条路径 + 截断 50% 再生 3 次 + 加权投票）
python harness.py --strategy prefix_consistency --dataset aqua --n_paths 3 --truncation_ratio 0.5 --regen_count 3

# Multi-Agent Debate（3 个 Agent × 2 轮辩论）
python harness.py --strategy multi_agent_debate --dataset aqua --n_agents 3 --n_rounds 2

# Step-Aware Verifier（3 条路径 + 每步验证）
python harness.py --strategy step_verifier --dataset aqua --n_paths 3
```

### 批量测试（50 条样本）

```bash
# 方式一：Python 脚本（串行，推荐）
python scripts/run_experiments.py

# 只跑单个策略
python scripts/run_experiments.py --strategy base_cot

# 并行运行（可能触发 API 限流）
python scripts/run_experiments.py --parallel --max_workers 3

# 方式二：Bash 脚本
bash scripts/run_all_50.sh
```

### 自定义参数

```bash
python harness.py \
  --strategy base_cot \
  --dataset aqua \
  --n_samples 100 \
  --model deepseek-v4-flash \
  --api_key "$DMX_API_KEY" \
  --base_url "https://www.dmxapi.cn/v1" \
  --temperature 0.7 \
  --max_tokens 1024 \
  --output_dir experiments/runs
```

---

## 📊 实验结果分析

### 对比多次实验

```bash
# 对比最近 5 次实验
python -m eval.analyze --runs_dir experiments/runs --latest 5

# 对比指定实验
python -m eval.analyze --runs_dir experiments/runs --run_ids 20260610_224644 20260610_225338
```

### Harness 子系统覆盖矩阵

```bash
python harness_report.py
```

输出每种策略对 Harness 五子系统（Instructions / Tools / Environment / State / Feedback）的覆盖情况。

---

## 🧠 六种 CoT 策略详解

### 1. Base COT

- **原理**：在 Prompt 中加入 "Let's think step by step"，引导模型逐步推理
- **参数**：无额外参数
- **Harness 覆盖**：Instructions + Environment（2/5）
- **特点**：最简单、最快、性价比最高

### 2. Self-Consistency

- **原理**：对同一问题采样多条推理路径，通过多数投票确定最终答案
- **参数**：`--n_paths N`（推理路径数，默认 3）
- **Harness 覆盖**：Instructions + Environment + State（3/5）
- **特点**：利用采样多样性减少偶然错误，但计算成本随路径数线性增长

### 3. Prefix Consistency

- **原理**：对每条 CoT 推理链在中间截断（如 50%），用前缀重新生成后续内容。正确推理链的原始答案在再生中复现率更高，用这个复现率作为权重进行加权多数投票
- **参数**：`--n_paths N`（初始路径数，默认 5），`--truncation_ratio R`（截断比例，默认 0.5），`--regen_count K`（每前缀再生次数，默认 3），`--weight_fn F`（权重函数：`linear`/`quadratic`/`cubic`/`unanimous`，默认 `linear`）
- **Harness 覆盖**：Instructions + Environment + State + Feedback（4/5）
- **特点**：无需 logprob 或自评 prompt，用轻量级的再生一致性作为可靠性信号。论文显示可达标准 Self-Consistency 平台准确率，token 消耗最高降低 21 倍（中位数 4.6 倍）

### 4. Step-Aware Verifier

- **原理**：生成多条推理路径后，对每条路径的每一步调用 Verifier 模型打分，筛选总分最高的路径
- **参数**：`--n_paths N`（推理路径数，默认 3）
- **Harness 覆盖**：Instructions + Tools + Environment + State + Feedback（5/5）
- **特点**：完整覆盖 Harness 五子系统，提供步骤级可解释性，但 API 调用量巨大、速度极慢

### 5. RAG + COT

- **原理**：在推理前从知识库检索相关数学知识，注入 Prompt 中辅助推理
- **参数**：`--top_k K`（检索文档数，默认 3）
- **Harness 覆盖**：Instructions + Tools + Environment + State（4/5）
- **特点**：当知识库质量高时加速推理，但低质量检索会引入干扰（retrieval noise）

### 6. Multi-Agent Debate

- **原理**：多个 LLM Agent 分别扮演不同角色（严谨分析者 / 创意解题者 / 怀疑批评者），进行多轮讨论与互评，最终投票决定答案
- **参数**：`--n_agents N`（Agent 数，默认 3），`--n_rounds R`（辩论轮数，默认 2）
- **Harness 覆盖**：Instructions + Environment + State + Feedback（4/5）
- **特点**：准确率最高，能通过多视角互评纠正单模型盲区，但 Token 消耗最大

---

## 🔧 Harness Engineering 设计思想

本项目借鉴 [walkinglabs/learn-harness-engineering](https://github.com/walkinglabs/learn-harness-engineering) 的五子系统模型：

| 子系统 | CoT 中的体现 | 激活策略 |
|---|---|---|
| **Instructions** | Prompt 模板设计（推理格式、角色指令、评分标准） | 全部 |
| **Tools** | 外部工具调用（检索器、Verifier 元推理工具） | rag_cot, step_verifier |
| **Environment** | AQuA 任务环境（数据加载、答案提取、评估指标） | 全部 |
| **State** | 运行时状态管理（多路径历史、检索上下文、辩论记录） | self_consistency, prefix_consistency, rag_cot, multi_agent_debate, step_verifier |
| **Feedback** | 反馈闭环（步骤级验证打分、多 Agent 互评纠错、前缀再生一致性） | prefix_consistency, multi_agent_debate, step_verifier |

**核心洞察**：子系统覆盖数 ≠ 准确率。`multi_agent_debate`（4/5）通过 Feedback 子系统实现了最高准确率（94%），而 `rag_cot`（4/5）因检索噪声导致准确率最低（78%）。`prefix_consistency`（4/5）以极轻量的 Feedback 机制（再生一致性）提供了介于 self_consistency 和 step_verifier 之间的性价比选择。

---

## 📈 50 样本实验结果（AQuA test 前 50 条）

| 策略 | 准确率 | 平均耗时/条 | 平均 Token | 综合性价比 |
|---|---|---|---|---|
| **multi_agent_debate** | **94.0%** ⭐ | 42.4s | 368.6 | ⭐⭐⭐⭐ 准确率最高 |
| base_cot | 92.0% | 5.2s | 174.4 | ⭐⭐⭐⭐⭐ **最佳性价比** |
| self_consistency | 92.0% | 16.1s | 189.1 | ⭐⭐⭐ 与 base 持平但更慢 |
| prefix_consistency | *待实验* | *待实验* | *待实验* | 预期：⭐⭐⭐⭐ 高效反馈 |
| step_verifier | 92.0% | **206.6s** | 387.9 | ⭐ 极慢，收益有限 |
| rag_cot | **78.0%** ▼ | 4.2s | 151.3 | ⭐⭐ 检索噪声损害性能 |

> 详细分析见 [`experiments/report_50_sample_20260611.md`](experiments/report_50_sample_20260611.md)

---

## 📝 核心文件说明

| 文件 | 说明 |
|---|---|
| `harness.py` | 实验主入口，注册全部策略，支持命令行参数，实时显示进度与准确率 |
| `scripts/run_experiments.py` | 50 样本批量测试 Python 脚本，支持串行/并行/单策略模式 |
| `scripts/run_all_50.sh` | 50 样本批量测试 Bash 脚本，顺序执行全部 6 种策略 |
| `eval/metrics.py` | 评估指标：准确率、推理步数、Token 估算、多实验对比 |
| `eval/analyze.py` | CLI 分析工具，对比多次实验并打印格式化表格 |
| `harness_report.py` | 生成 Harness 子系统覆盖矩阵 |
| `scripts/init.sh` | 环境验证脚本，检查依赖、模块导入、各策略干跑 |
| `feature_list.json` | 10 项功能的状态追踪（全部已完成） |
| `docs/progress.md` | 开发进度日志 |

---

## ⚠️ 已知问题与注意事项

1. **API 限流**：DMXAPI 不支持 `n > 1` 的批量生成，`self_consistency`、`prefix_consistency` 和 `step_verifier` 已改为循环调用 `n=1`
2. **step_verifier 极慢**：50 样本约需 3 小时，因每步都需额外 API 调用
3. **prefix_consistency 速度**：50 样本约需 1.5 小时（3 路径 × 3 再生 = 9 倍调用），但 token 效率优于标准 Self-Consistency
4. **rag_cot 检索噪声**：当前 keyword-based 检索可能引入无关知识，建议在高质量知识库上使用
5. **Windows 编码**：如遇到 GBK 解码错误，确保文件使用 UTF-8 编码

---

## 📚 参考文献

- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models: https://arxiv.org/abs/2201.11903
- Self-Consistency Improves Chain of Thought Reasoning in Language Models: https://arxiv.org/abs/2203.11171
- Prefix Consistency (PC-WMV): https://arxiv.org/abs/2605.07654
- RAG + COT (IRCoT): https://arxiv.org/abs/2212.09095
- Multi-Agent Debate: https://arxiv.org/abs/2305.14325
- Step-Aware Verifier: https://arxiv.org/abs/2310.15123
- Harness Engineering: https://github.com/walkinglabs/learn-harness-engineering
- AQuA Dataset: https://github.com/deepmind/AQuA

---

## 🤝 贡献与使用

本项目为思维链推理的实验性实现，欢迎基于本项目进行扩展：

- 接入更多数据集（GSM8K、MATH 等）
- 优化 RAG 检索器（使用 Embedding-based 语义检索）
- 改进 Step Verifier（稀疏验证、本地轻量模型）
- 尝试更多 CoT 变体（Tree-of-Thought、Program-Aided LLM 等）
- 探索 Prefix Consistency 的变体（不同截断比例、基于句子的截断、多截断点融合）

---

> **Co-Authored-By**: Claude <noreply@anthropic.com>
