# APO

## 1. 现状

### 1.1 学通信图

- G-Designer (ICML'25 spotlight) — VAE 把 (task, agents) 编码 → 解码出一张 sparse adjacency matrix
- GPTSwarm (ICML'24 oral) — 节点固定，REINFORCE 学边权 + 每节点 prompt
- AgentPrune (ICLR'25) — 从 full graph 剪枝
- EIB-LEARNER (EMNLP'25) — causal framework 平衡 "error suppression" 与 "insight propagation"，得到 moderately-sparse 拓扑
- Graph-GRPO (arxiv 2603.02701, 2026) — 把 GRPO 用到 edge optimization，提出 edge-level marginal success rate；6-bench SOTA 92.45%
- GoAgent (arxiv 2603.19677, 2026) — 把"协作小组"当原子节点，autoregressive 构图 + 信息瓶颈
- OFA-MAS (arxiv 2601.12996, 2026) — Mixture-of-Experts graph generative model 跨域生成
- CARD (ICLR'26) — conditional VAE，topology 随环境信号 (model upgrade / tool change) 变
- HeGFlow — heterogeneous graph 把 agent / tool / step 都建模
- CayleyTopo (arxiv 2604.09703, 2026) — 数论先验 Cayley graph
- MASFactory (arxiv 2603.06007, 2026) — graph-centric "Vibe Graphing" 框架
- MACNET — 千 agent DAG，发现 "collaborative scaling law"

### 1.2 学操作组合

- MaAS (ICML'25 oral) — agentic supernet，按 query 从 8 个预设 operator (CoT / Self-Consistency / Debate / Self-Refine / MultiPersona / LLM-Blender / Self-RAG / Exit) 采样组合；6-bench SOTA，成本只占 baseline 的 6-45%
- MASS (Google, arxiv 2502.02533, 2025) — 三阶段交替优化：block-level prompt → topology → workflow-level prompt
- AgentSquare (ICLR'25) — 把 agent 内部拆 4 module (Planning / Reasoning / Tool / Memory)，evolutionary search 1050 种组合
- ARG-Designer (AAAI'26 oral) — LLM autoregressive 一步一步吐 `(role, edges)`，纯 SFT，没 RL

### 1.3 学完整 workflow / code

- AFlow (ICLR'25 oral) — workflow 当 Python code，MCTS 在 code 空间搜；GSM8K 78.5%，成本 4.55% of GPT-4o
- ADAS / Meta Agent Search (ICLR'25, NeurIPS'24 best paper) — LLM 自己写 code 当 agent，archive 不断扩展，理论 Turing-complete
- MAS-GPT (ICML'25) — train 一个 LLM 一次性生成完整 MAS code
- FlowReasoner (arxiv 2504.15257, 2025) — distilled from R1 + RL，query-level meta-agent
- MAS² / MAS-Orchestra (ICLR'26) — recursive self-generation / function-calling RL

### 1.4 学 agent 数量 / 退场

- DyLAN (COLM'24) — 跑一遍后用 importance score 删次要 agent
- AgentDropout (arxiv 2503.18891, 2025) — dynamic agent elimination
- AutoAgents — LLM 临时生成需要的 agent（学增加方向，不学减）

### 1.5 inference-time / training-free

- EvoMAC (ICLR'25) — test-time 用 textual backpropagation 改图
- MAS-ZERO (2026) — inference-time 自演化，no validation set
- EvoMAS / AdaptOrch (2026) — evolutionary / orchestration topology selector

---

## 2. 共同的局限

5 个 design dimension 没有一个被全部学到：

| Dimension | 谁学了 |
|---|---|
| Agent 数量 | DyLAN / AgentDropout |
| 角色分配 | AutoAgents / MAS-GPT |
| 通信图 | G-Designer / GPTSwarm / Graph-GRPO |
| 调度顺序 | ARG-Designer (autoregressive) |
| 何时停 | MaAS 有 exit operator |

---

## 3. 三类深层问题

### 3.1 表达力局限 —— 每个工作只学架构的一小角

没人能从原始元素 (agent / role / edge / 顺序) 端到端学一个完整架构分布。

### 3.2 优化算法局限 —— 信号粗糙

- 早期 (G-Designer, GPTSwarm) 用 single-sample REINFORCE，方差大。
- 2026 Graph-GRPO 在 edge 上做 marginal success rate 解决了这个问题（"含 edge (i,j) 的成功率"作 group-relative advantage）。
- 但 Graph-GRPO 也只在 edge 一维做了 fine-grained credit，gates / role / sequence 都还没有相应机制。

### 3.3 执行模型局限 —— agent 太薄

- 几乎所有工作 "1 agent = 1 次 LLM call"。
- 聚合普遍靠 "再调一次 LLM 写 summary"（多花成本 + 容易引入 bias）。
- 终止靠固定 cycle / round 数（外部超参），不交给模型学。

---

## 4. APO vs 25+ 个工作 —— 一张差异化表

| | 学 agent? | 学 role? | 学 edge? | 学 seq? | Inner loop | 优化 |
|---|---|---|---|---|---|---|
| GPTSwarm | × | × | edge weight | × | × | REINFORCE |
| AFlow | code | code | code | code | optional | MCTS |
| AgentSquare | × | module | × | × | module | evolutionary |
| G-Designer | × | × | one graph | × | × | REINFORCE |
| MaAS | operator | operator | operator | × | operator | REINFORCE+text |
| ARG-Designer | autoregressive | AR | AR | × | × | SFT only |
| ADAS | code | code | code | code | optional | meta-search |
| AgentPrune / Dropout | dropout | × | prune | × | × | REINFORCE |
| EIB-LEARNER | × | × | edge causal | × | × | REINFORCE |
| **Graph-GRPO** | × | × | **edge GRPO** | × | × | edge-level GRPO |
| **APO (ours)** | **gate Bern** | **role Cat (8)** | **latent + SBM** | **PL** | **ReAct + tools** | **shaped-advantage GRPO** |

可直接对比的相关工作包括 Graph-GRPO（2026），其算法路线同样使用 GRPO。差异点：

- Scope: Graph-GRPO 只学 edge (N×N)。APO 学 gates + roles + edges + seq。

---

## 5. Benchmark

MATH-500：MATH 数据集 500 题代表，竞赛级数学. AFlow MATH 69.2, MaAS ≈75. 测难度跳跃后我们的 prior 是否仍有效.

DROP：9k 段落 + 离散数值推理 (count, sort, arith). AFlow 68.3, ADAS 用过. 测 reasoning 跨 family transfer.

HotpotQA：multi-hop QA, 需要跨段落推理. AFlow 68.9. 测 Researcher / Planner role 是否真有用.
