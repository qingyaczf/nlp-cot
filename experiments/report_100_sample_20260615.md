# 100 样本实验报告（AQuA test 前 100 条）

**Date:** 2026-06-15  
**Model:** deepseek-v4-flash
**Dataset:** AQuA test split（前 100 条）  
**API Endpoint:** https://api.deepseek.com/v1

---

## 1. 实验概览

本次实验在 AQuA 测试集前 100 条样本上运行了 5 种 CoT 策略，评估指标包括准确率、平均输出 token 数、平均推理步数。

| 策略 | Run ID | 准确率 | 正确/总数 | 平均输出 Token | 平均输入 Token | 平均推理步数 |
|---|---|---|---|---|---|---|
| base_cot | 20260615_221801 | **91.0%** | 91/100 | 187.6 | 130.0 | 5.8 |
| self_consistency | 20260615_121728 | **94.0%** | 94/100 | 238.6 | 130.0 | 8.0 |
| rag_cot | 20260615_224513 | **78.0%** | 78/100 | 175.7 | 201.3 | 5.2 |
| multi_agent_debate | 20260615_231927 | **91.0%** | 91/100 | 370.8 | 0.0* | 8.3 |
| step_verifier (LLM) | 20260614_232143 | **94.0%** | 94/100 | 563.7 | 0.0* | 21.6 |

> \* `multi_agent_debate` 与 `step_verifier` 的输入 token 因多轮交互统计方式不同，当前记录为 0。

---

## 2. 关键发现

### 2.1 准确率梯队

- **第一梯队（94%）**：`self_consistency`、`step_verifier (LLM)`
  - Self-Consistency 以极低的额外 token 开销（仅比 base_cot 高 27%）达到了最高准确率，性价比最优。
  - Step-Verifer (LLM) 虽然准确率同样为 94%，但输出 token 高达 563.7，是 Self-Consistency 的 2.4 倍，成本效益较低。

- **第二梯队（91%）**：`base_cot`、`multi_agent_debate`
  - Base COT 以最简单的逻辑实现了 91% 的准确率，输出 token 最低（187.6），是**基础性价比之王**。
  - Multi-Agent Debate 消耗了最多的输出 token（370.8），但准确率仅为 91%，低于其 50 样本时的 94%，说明在扩大样本量后，多 Agent 辩论的稳定性有所下降。

- **第三梯队（78%）**：`rag_cot`
  - RAG+COT 准确率最低，与 50 样本结果持平（78%）。检索噪声问题在更大样本集上依然显著，当前 keyword-based 检索器未能带来正向收益。

### 2.2 Token 消耗对比

| 策略 | 平均输出 Token | 相对 base_cot 倍数 |
|---|---|---|
| base_cot | 187.6 | 1.0× |
| rag_cot | 175.7 | 0.94× |
| self_consistency | 238.6 | 1.27× |
| multi_agent_debate | 370.8 | 1.98× |
| step_verifier (LLM) | 563.7 | 3.00× |

**结论**：
- `rag_cot` 的 token 消耗最低，但准确率也最低，说明检索内容未能有效辅助推理。
- `self_consistency` 以 1.27 倍的 token 代价换取了 3 个百分点的准确率提升（91% → 94%），投资回报最高。
- `multi_agent_debate` 和 `step_verifier` 的 token 消耗巨大，但准确率并未显著优于 `self_consistency`。

### 2.3 推理步数观察

- `step_verifier` 的平均推理步数高达 **21.6**，远超其他策略（base_cot 5.8、self_consistency 8.0），这是因为 verifier 会对每条路径进行逐步评估，导致输出极度冗长。
- `multi_agent_debate` 的 8.3 步与 `self_consistency` 的 8.0 步接近，但 token 消耗高出 55%，说明多 Agent 对话中的冗余表述较多。

---

## 3. 与 50 样本结果对比

| 策略 | 50 样本准确率 | 100 样本准确率 | 变化 |
|---|---|---|---|
| base_cot | 92.0% | 91.0% | -1.0% ↓ |
| self_consistency | 94.0% | 94.0% | 持平 |
| rag_cot | 78.0% | 78.0% | 持平 |
| multi_agent_debate | 94.0% | 91.0% | -3.0% ↓ |
| step_verifier (LLM) | 92.0% | 94.0%* | +2.0% ↑ |

> \* step_verifier 100 样本由 deepseek-v4-flash 运行，与 50 样本的 deepseek-chat 结果存在模型差异，不可直接比较。

**分析**：

- `self_consistency` 和 `rag_cot` 在扩大样本量后表现稳定，说明策略的鲁棒性较好。
- `multi_agent_debate` 准确率从 94% 降至 91%，可能原因：多 Agent 辩论在更复杂的题目上容易出现"过度讨论"或"从众效应"，导致错误答案被巩固。
- `base_cot` 轻微下降 1%，属于正常波动范围。

---

## 4. Harness 子系统覆盖与准确率关系（100 样本）

| 策略 | 子系统覆盖 | 100 样本准确率 |
|---|---|---|
| base_cot | 2/5 (Instructions + Environment) | 91.0% |
| rag_cot | 4/5 (+ Tools + State) | 78.0% |
| self_consistency | 3/5 (+ State) | 94.0% |
| multi_agent_debate | 4/5 (+ State + Feedback) | 91.0% |
| step_verifier | 5/5 (全部) | 94.0% |

**核心洞察**：
- 子系统覆盖数与准确率**无单调正相关**。`rag_cot`（4/5）准确率最低（78%），说明 Tools 子系统的质量比覆盖本身更关键。
- `self_consistency`（3/5）以最少的外围机制达到了最高准确率，证明**高质量的局部评估**（路径质量评分）比复杂的系统架构更高效。
- `multi_agent_debate`（4/5）在 100 样本上未能保持 50 样本的领先优势，说明 Feedback 子系统在多轮交互中可能引入噪声。

---

## 5. 待补全实验

以下策略尚未完成 100 样本测试（deepseek-chat）：

- [ ] **prefix_consistency**：50 样本准确率 94.0%，预计 100 样本表现稳定
- [ ] **few_shot_cot**：尚未进行大规模基准测试
- [ ] **step_verifier（本地 DeBERTa）**：50 样本准确率 94.0%，本地验证速度快、零 API 费用
- [ ] **step_verifier（deepseek-chat LLM）**：当前仅有 deepseek-v4-flash 结果

---

## 6. 实验文件索引

| Run ID | 策略 | 样本数 | 模型 |
|---|---|---|---|
| 20260615_221801 | base_cot | 100 | deepseek-chat |
| 20260615_121728 | self_consistency | 100 | deepseek-chat |
| 20260615_224513 | rag_cot | 100 | deepseek-chat |
| 20260615_231927 | multi_agent_debate | 100 | deepseek-chat |
| 20260614_232143 | step_verifier | 100 | deepseek-v4-flash |

---

> **Co-Authored-By**: Claude <noreply@anthropic.com>
