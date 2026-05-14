# APO Method

## 1. 整体

```
task → head (Qwen3-0.6B + 4 typed heads) → 4 typed logits
    → sample_arch → ConcreteArch
    → executor: PL-permutation 调度 + ReAct agents + Synth
    → reward
    → GRPO (typed log_pi + entropy, no KL) → 反传到 head
```

只 head 训练。worker (agent + synth) 是 OpenAI-compatible API，不动。

---

## 2. Head：latent agent embedding + 4 typed heads

```
backbone(task) → context h ∈ R^{d_h}
   ↓ agent_proj (Linear: d_h → N·d) + slot_emb (Embedding(N, d))
U ∈ R^{N × d}                       # 每个 agent slot 的 latent

gate_logits[i]    = w_g · U[i]                            (Bern)
role_logits[i, r] = (W_Q · U[i])[r]                       (Cat-7)
edge_logits[i,j]  = U[i]^T M U[j] / √d                    (latent space)
                  + softmax(role_logits)[i]^T B softmax(role_logits)[j]   (SBM)
                  + b0
seq_scores[i]     = w_s · U[i]                            (PL)
```

4 个 head 共享 `U[i]` —— "slot i 该是什么 agent" 是一个整体语义，不该被拆成 4 个无关输出。

Edge 用 latent + SBM 双源：
- `U M U^T` = Latent Space Network model (Hoff'02), agent 对的"个性 affinity"
- `Q^T B Q` = Stochastic Block Model (Holland'81), role 对的"该不该连"
- 训练完后 **B 矩阵 (7×7) 可读**：直接 heatmap 看出 "Critic→Solver=0.9 / Solver→Solver=0.2"，是 paper 的可解释性卖点

参数量 ~100K 可训练，backbone frozen。

---

## 3. 4 typed distributions 的采样

| Component | 分布 | 怎么采 |
|---|---|---|
| gates | Bernoulli(σ(gate_logits)) | per-slot 独立；至少 1 个强制 active |
| roles | Categorical(softmax(role_logits)) | per-slot multinomial（仅 active 有意义） |
| edges | Bernoulli(σ(edge_logits)) | mask 到 active pair, 去对角 |
| sequence | Plackett-Luce(seq_scores) | 见 §4 |

`sequence` 是 active slot 的一个排列，长度 = #active。

---

## 4. Plackett-Luce

为什么不直接 N 个 priority + argsort：argsort 不可导，group 内 G 个 sample 微小不同的 score 可能 argsort 出**完全一样**的 order，advantage 没法 attribute。

PL 解决：依次 categorical without replacement，每步 vocab 是剩下的 slot：

\[
P(i_t \mid i_{<t}) = \frac{e^{s_{i_t}}}{\sum_{j \in \text{remaining}_t} e^{s_j}}
\]

\[
\log P(\pi \mid s) = \sum_{t=1}^{K} \left[ s_{i_t} - \log \sum_{j \in \text{remaining}_t} e^{s_j} \right]
\]

跟 LLM next-token CE 同形，vocab 动态收缩。SFT teacher 的 perm 当 token target、GRPO 的 sampled perm 当 token 序列 —— 同一公式两用。

---

## 5. Executor

```
for big_round in 1 .. safety_max_big_rounds=20:
  for slot in arch.sequence:        # PL permutation
    incoming = msgs from slots with edge → slot
    agent[slot].run(task, incoming)
  
  verdict = synth.judge(task, transcript)
  if verdict.is_done: return verdict.answer
```

### 5.1 Agent = ReAct inner loop

```
for inner_round in 1 .. 8:
  reply = worker.chat(role_prompt, task + incoming + scratchpad)
  if reply has "ACTION: <tool>":
    obs = call_tool(...)
    scratchpad += reply + "\nOBSERVATION:\n" + obs
  else:
    return reply
```

3 个 tool: `python_exec` (subprocess + 5s timeout) / `sympy_check` (sympy) / `web_search` (stub)。所有 role 都能用所有 tool。

### 5.2 Synth

```
SYNTH_PROMPT:
  "Output EXACTLY one of:
     ANSWER: <X>
     CONTINUE
   DO NOT reason."
```

DeepSeek API call，prompt 严格限制只能输 ANSWER:/CONTINUE。

错判被 reward 自然修正：早 done 但答案错 → reward 0 → 学到下次别早 done；CONTINUE 太久 → cost 扣分。

### 5.3 Safety cap

`safety_max_big_rounds=20`, `safety_max_inner_rounds=8`, `safety_max_tokens_per_call=2048`。**不是 head 输出参数** —— reward cost 项 (`-λ_c · #calls`) 教 head 提前停。

---

## 6. SFT loss

\[
\mathcal{L}_{\text{SFT}} = \mathcal{L}_g^{\text{BCE}} + \mathcal{L}_Q^{\text{CE}} + \mathcal{L}_e^{\text{BCE}} + \mathcal{L}_s^{\text{PL-NLL}}
\]

| 分量 | mask |
|---|---|
| Gate BCE | 全 N |
| Role CE | 仅 active slot |
| Edge BCE | 仅 active pair, 去对角 |
| Seq PL-NLL / K | K = #active |

Teacher = 73 个手编 NamedArch（`architecture/library.py`）。每 epoch reshuffle pairing，逼模型学"好架构 region"而不是"task A → arch X"死记。

没有 KL：head 不是 LM，没有"语言能力"要保护。

---

## 7. GRPO loss

```
for batch of tasks:
  head_out = head(tasks)
  for g = 1..G=4:
    arch_g = sample_arch(head_out)
    trace = executor.run(task, arch_g)
    reward[g] = correctness - λ_n·#agents - λ_e·#edges - λ_c·#calls
  A[g] = (reward[g] - mean) / (std + ε)
  log_pi[g] = log_Bern(gates) + log_Cat(roles) + log_Bern(edges) + log_PL(seq)
  loss = -mean(A * log_pi) - α · H(head_out)
```

\[
\log \pi_\phi(a \mid x) = \sum_i \log \text{Bern}(g_i) + \sum_{i:\text{active}} \log \text{Cat}(r_i) + \sum_{(i,j):\text{active pair}} \log \text{Bern}(e_{ij}) + \log \text{PL}(\sigma; s)
\]

`#calls` 自动包含 ReAct inner + Synth → reward 教 head 选不会反复调工具的架构。

Entropy bonus（4 个分布之和）替代 KL 防 mode collapse。`α = 0.01` 默认。

---

## 8. Typed-component-level credit assignment

> Borrow + 推广 Graph-GRPO (2026)。当前是 graph-level，要升级。

Graph-GRPO 在 edge 上算 marginal success rate：

\[
S_{ij}^{(e)} = \frac{\sum_k \mathbb{1}[(i,j) \in E^{(k)}] \cdot r_k}{\sum_k \mathbb{1}[(i,j) \in E^{(k)}] + \epsilon}
\]

含义："含 edge (i,j) 的那些 group sample 的平均 reward"。advantage = `(S - μ) / σ`。

我们推广到 4 个 components：

\[
S_i^{(g)} = \frac{\sum_k \mathbb{1}[g_i^{(k)}=1] \cdot r_k}{\sum_k \mathbb{1}[g_i^{(k)}=1] + \epsilon}
\]

\[
S_{i,r}^{(Q)} = \frac{\sum_k \mathbb{1}[r_i^{(k)}=r] \cdot r_k}{\sum_k \mathbb{1}[r_i^{(k)}=r] + \epsilon}
\]

\[
S_{t,j}^{(\sigma)} = \frac{\sum_k \mathbb{1}[\sigma^{(k)}_t = j] \cdot r_k}{\sum_k \mathbb{1}[\sigma^{(k)}_t = j] + \epsilon}
\]

每个 component 各自有 advantage、各自的 log_prob 各自被 weighted：

\[
\mathcal{L} = -\sum_{c \in \{g, Q, e, \sigma\}} \sum_i A_i^{(c)} \cdot \log \pi^{(c)}_i - \alpha \mathcal{H}
\]

Graph-GRPO 在 edge 上从 graph-level 升 edge-level 拿到 +1.82%（他们 Table 2）。我们 4 个 components 都升，预期 +2~3%。
