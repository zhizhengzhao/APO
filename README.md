# APO — Architecture Policy Optimization

Learn a task-conditioned **distribution over multi-agent architectures** for LLMs.

![A multi-agent architecture is determined by 4 things](assets/fig1_four_things.png)

The head reads a task with a small LM (Qwen3-0.6B by default), emits the
parameters of an architecture distribution. We sample a concrete (discrete)
architecture from this distribution, run it with a worker LLM (DeepSeek-Chat
API by default), and use the task reward to optimize the head.

The head is the **only** thing we train. Workers are off-the-shelf API calls.

---

## v3 design (current)

### 1. 4 typed distributions, not one Gaussian

The head outputs **4 independent typed tensors**, each with its own natural
distribution and loss:

| Output | Shape | Distribution | SFT loss |
|---|---|---|---|
| `gate_logits` | `[N]` | Bernoulli per slot | BCE |
| `role_logits` | `[N, R]` | Categorical per slot (R=7) | CE |
| `edge_logits` | `[N, N]` | Bernoulli per pair (latent + SBM) | BCE |
| `seq_scores`  | `[N]` | Plackett-Luce permutation | listwise NLL |

Sampling: gates and edges are independent Bernoullis, roles a per-slot
categorical, sequence a Plackett-Luce permutation **over the active slots**
(no repeats; length = #active).

### 2. Latent agent embedding (the key inductive bias)

```
backbone(task) → context  ∈ R^{H}
                  ↓ MLP body
                h         ∈ R^{d_h}
                  ↓ agent_proj + slot_emb
                U         ∈ R^{N × d}        (per-agent latent embedding)
                  ↓
        ┌─ gate_logits[i]   = w_g · U[i]
        ├─ role_logits[i,r] = (W_Q · U[i])[r]
        ├─ edge_logits[i,j] = (U[i]^T M U[j]) / √d  +  Q[i]^T B Q[j]  +  b0
        └─ seq_scores[i]    = w_s · U[i]
```

All 4 heads share the per-slot latent **U[i]** — "what kind of agent goes in
slot i" is one concept, not 4 disjoint outputs. The edge formula combines a
**Latent Space Network model** (Hoff 2002) with a **Stochastic Block Model**
on roles (Holland 1981); the **B matrix is directly readable** ("does Critic
listen to Solver?").

### 3. ReAct agents + tools

Each active slot becomes an `Agent` running a **ReAct inner loop**:

```
agent.run(task, incoming_msgs):
  for inner_round = 1..safety_cap:
    reply = worker.chat(role_prompt, accumulated_scratchpad)
    if reply has ACTION: <tool>:
      observation = call_tool(...)
      append to scratchpad; continue
    else:
      return reply               # final reply for this turn
```

3 tools available to all roles: `python_exec`, `sympy_check`, `web_search`
(stub). The reward function's cost term naturally teaches the system to use
tools sparingly.

### 4. Synth (DONE/CONTINUE judge) replaces aggregator

After each big round, a single LLM call (`Synth`) reads the transcript and
outputs **exactly one of**:

```
ANSWER: <the final answer>
CONTINUE
```

Synth does **not** reason. If it says ANSWER, we're done; if CONTINUE, we run
the sequence one more time. Hard safety cap = 20 big rounds — never reached
in practice because the reward's cost term penalizes long discussions.

### 5. No KL, no Gaussian

The head is a fresh policy planner — not a language model — so there's no
"language ability" to protect with KL. GRPO uses a small **entropy bonus**
on the typed distributions to prevent premature mode collapse.

---

## Layout

```
src/arch_policy/
  config.py                # ArchSpec / ModelSpec / TrainSpec
  architecture/
    spec.py                # ArchLogits / ArchTargets typed dataclasses
    sampler.py             # sample_arch + 4 typed log_prob_* + PL
    library.py             # 30+ NamedArch prototypes (7 roles)
    encoder.py             # NamedArch → ArchTargets (typed)
  head/
    model.py               # Latent agent embedding + 4 typed heads
  executor/
    prompts.py             # 7 role prompts + ReAct + Synth prompt
    tools.py               # python_exec / sympy_check / web_search
    agent.py               # Agent: ReAct inner loop
    synth.py               # Synth: ANSWER:/CONTINUE judge + heuristic fallback
    multi_agent.py         # MultiAgentExecutor (Worker abstraction + main loop)
    openai_worker.py       # DeepSeek / OpenAI-compatible worker
  data/
    tasks.py               # GSM8K / MATH / HumanEval / synthetic loaders
    sft_data.py            # task ↔ random ArchTargets dataset
  reward/
    grade.py               # numeric / boxed-MATH / HumanEval-exec graders
    compute.py             # correctness − costs
  training/
    sft.py                 # train_sft + 4 typed losses
    grpo.py                # train_grpo + typed log_pi + entropy bonus
  baselines.py             # 10 fixed-topology baselines

scripts/
  00_setup_env.sh          # conda env + deps + editable install
  01_download_models.py    # pre-fetch Qwen3-0.6B (head)
  02_smoke_test.py         # 24 CPU-only sanity tests
  03_run_sft.py            # Stage-1 SFT runner
  04_inspect_head.py       # load checkpoint, print sample architectures
  05_evaluate.py           # baseline / head eval against GSM8K/MATH/HumanEval
  06_run_grpo.py           # Stage-2 GRPO runner

tests/
  test_architecture.py     # 11 tests: typed logits, sampler, PL, encoder, log_prob
  test_executor.py         # 11 tests: each baseline, ReAct, Synth, heuristic
  test_sft_step.py         #  2 tests: typed loss + dummy backbone descent

configs/
  default.yaml             # mirror of dataclass defaults
```

---

## Math: training + GRPO

### SFT (Stage 1)

\[
\mathcal{L}_{\text{SFT}} = w_g \mathcal{L}_{\text{BCE}}(\text{gates}) + w_Q \mathcal{L}_{\text{CE}}(\text{roles}) + w_e \mathcal{L}_{\text{BCE}}(\text{edges}) + w_s \mathcal{L}_{\text{PL-NLL}}(\text{seq})
\]

PL NLL of teacher permutation \(\pi^* = (i_1, \ldots, i_K)\):

\[
\mathcal{L}_{\text{PL-NLL}} = -\sum_{t=1}^{K} \left[ s_{i_t} - \log \sum_{j \in \text{remaining}_t} e^{s_j} \right]
\]

Each step is a categorical CE over the *remaining* (yet-to-be-picked) slots.
Same form as LLM next-token CE, with a dynamically shrinking vocabulary.

### GRPO (Stage 2)

Joint log-probability under the typed distributions:

\[
\log \pi_\phi(a \mid x) = \sum_i \log \text{Bern}(g_i; p^g_i) + \sum_i \log \text{Cat}(r_i; \pi^r_i) + \sum_{i,j} \log \text{Bern}(e_{ij}; p^e_{ij}) + \log \text{PL}(\sigma; s)
\]

Group-relative advantage with `G` samples per task:

\[
\hat A_g = \frac{R_g - \frac{1}{G}\sum_{g'} R_{g'}}{\text{std}(R) + \epsilon}
\]

Loss (no KL):

\[
\mathcal{L}_{\text{GRPO}} = -\frac{1}{G}\sum_g \hat A_g \cdot \log \pi_\phi(a_g \mid x) - \alpha \cdot \mathcal{H}[\pi_\phi]
\]

Reward:

\[
R = \mathbb{1}[\hat y = y^*] - \lambda_n \cdot |a|_{\text{agents}} - \lambda_e \cdot |a|_{\text{edges}} - \lambda_c \cdot |\tau|_{\text{LLM calls}}
\]

`|τ|_LLM_calls` includes inner ReAct rounds + Synth calls — that's how the
system is incentivized to be efficient.

---

## Getting started

### 1. Install

```bash
cd /path/to/arch_policy
bash scripts/00_setup_env.sh
conda activate arch_policy
```

(`scripts/00_setup_env.sh` writes a conda activate hook setting
`HF_ENDPOINT=https://hf-mirror.com` for users behind GFW.)

### 2. CPU-only smoke test (no GPU, no API)

```bash
python scripts/02_smoke_test.py
# Expect: 24 PASS, 0 FAIL
```

### 3. Pre-download head model

```bash
python scripts/01_download_models.py
# Downloads Qwen3-0.6B (~1.2 GB)
```

### 4. Configure DeepSeek API for worker calls

```bash
export OPENAI_API_KEY="sk-...your-deepseek-key..."
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
```

### 5. Stage-1 SFT (a few hours on a single GPU)

```bash
python scripts/03_run_sft.py \
  --tasks_source gsm8k --n_tasks 1500 \
  --epochs 5 --batch_size 8 --device cuda:0 \
  --out_dir checkpoints/sft_v1
```

### 6. Inspect a mid-training checkpoint

```bash
python scripts/04_inspect_head.py --ckpt checkpoints/sft_v1/head_step100
# Shows gate_prob, role argmax, sampled architecture for 4 demo tasks
```

### 7. Evaluate on GSM8K

```bash
python scripts/05_evaluate.py \
  --mode head --head_ckpt checkpoints/sft_v1/head_step200 \
  --dataset gsm8k --n 200 \
  --worker openai --worker_model deepseek-chat \
  --out_jsonl results/apo_gsm8k.jsonl
```

Compare against the 10 baselines:

```bash
for bl in single solver_verifier chain_3 mesh_3 debate_3 self_consistency_5 \
          plan_solve_verify solver_critic_verifier tool_solver_verifier star_3; do
  python scripts/05_evaluate.py --mode baseline --baseline $bl \
    --dataset gsm8k --n 200 --worker openai --worker_model deepseek-chat \
    --out_jsonl results/${bl}_gsm8k.jsonl
done
```

### 8. Stage-2 GRPO

```bash
python scripts/06_run_grpo.py \
  --head_ckpt checkpoints/sft_v1/head_step100 \
  --dataset gsm8k --n 200 \
  --worker openai --worker_model deepseek-chat \
  --G 4 --epochs 2 \
  --out_dir checkpoints/grpo_v1
```

---

## What's *not* learned

By design, the following are **not** policy parameters:

- **`max_new_tokens`** per LLM call (engineering value)
- **`safety_max_big_rounds`** = 20 (engineering safety cap; reward cost teaches early stop)
- **`safety_max_inner_rounds`** = 8 (same)
- **Tool registry** (a fixed list; head doesn't choose which tools exist)

The reward function's cost terms (`-λ_n·#agents - λ_e·#edges - λ_c·#calls`)
are what shape the policy toward efficient choices.

---

## v3 vs v2 (in case you've seen the old code)

- **Head output**: flat 1311-dim Gaussian → **4 typed tensors**
- **Loss**: NLL on one big Gaussian + KL → **typed (BCE / CE / BCE / PL-NLL)**, no KL
- **GRPO log_pi**: diagonal Gaussian → **typed Bern + Cat + Bern + PL**
- **Sequence**: `[m_max, N]` softmax (with repeats) → **`[N]` PL permutation** (1 turn per active per round)
- **Aggregator LLM**: writes a final answer → replaced by **Synth** (DONE/CONTINUE judge)
- **Big rounds**: fixed `k_repeat_max=3` → **dynamic, terminated by Synth**
- **Agent**: 1 LLM call → **ReAct inner loop with tools**
- **Roles**: 4 (Solver/Critic/Verifier/ToolUser) → **7** (+Planner / Refiner / Researcher)
- **Worker**: local Qwen3-4B → **DeepSeek API** (head is the only thing trained)

---

## Cost ballpark

DeepSeek-Chat: $0.27 / M input tokens, $1.10 / M output tokens.

Per GSM8K question with a 3-agent chain:
- ~3 agent calls + 1 synth call = 4 LLM calls
- ~600 input tokens, ~200 output tokens per call
- ~$0.001 per question
- **GSM8K eval (200 questions, 10 baselines + APO + 5 ablations) ≈ $5**
- **GRPO training (200 tasks × 4 archs × 5 epochs) ≈ $20**
