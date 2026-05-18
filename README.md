# APO — Architecture Policy Optimization

学一个 task-conditioned 的多智能体架构分布。**Head = Qwen3-4B 换 4 个 typed 概率分布头**（输出架构参数）；**Worker = DeepSeek V4-Pro API**（跑所有 agent 和 Synth）。本地只训练 head。

Head backbone 默认 trainable（全参数 fine-tune via LoRA），三种模式：

| 模式 | flag | trainable params (Qwen3-4B) | 单卡 mem | 适合场景 |
|---|---|---|---|---|
| LoRA（默认） | `--lora_rank 32` | ~60M | 24GB+ | 推荐：cheap、防 overfit |
| 全参数 fine-tune | `--no-freeze_backbone --lora_rank 0` | ~4B | 80GB + GC | 大数据时 |
| 冻 backbone | `--freeze_backbone` | ~2M typed heads | 任何卡 | head-only baseline |

![A multi-agent architecture is determined by 4 things](assets/fig1_four_things.png)

## 文档

- [`STORY.md`](STORY.md) — 现有工作 + 局限 + 我们的位置
- [`METHOD.md`](METHOD.md) — head / sampler / executor / 训练公式

## 安装

```bash
bash scripts/00_setup_env.sh        # conda env + deps + editable install
conda activate arch_policy
python scripts/02_smoke_test.py     # 26 PASS, no GPU / no API
python scripts/01_download_models.py  # Qwen3-4B (~8 GB) — set HEAD_MODEL=Qwen/Qwen3-0.6B for smoke
```

配 worker（DeepSeek 或任何 OpenAI-compatible）：

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
```

## 跑

每个 epoch 都会自动存一个 `head_epoch{N}/` checkpoint；选其中表现最好的一个推到 GRPO。

```bash
# === Stage 1 SFT (本地) ============================================
# 默认 family-stratified 采样, tier_ratio (0.73/0.16/0.11), 每 epoch 存 ckpt
# 默认 trainability mode = full fine-tune (V3.5)

# (A) 推荐：LoRA, 单卡 24GB+ 都行，避免 overfit
python scripts/03_run_sft.py \
  --tasks_source mixed --epochs 5 \
  --lora_rank 32 \
  --device cuda:0 --out_dir checkpoints/sft

# (B) 全参数 fine-tune (重，9B 单卡 80GB + 必开 gradient checkpointing)
python scripts/03_run_sft.py \
  --tasks_source mixed --epochs 5 \
  --gradient_checkpointing \
  --device cuda:0 --out_dir checkpoints/sft

# (C) V3 baseline：只训 4 个 typed heads, backbone 冻死
python scripts/03_run_sft.py \
  --tasks_source mixed --epochs 5 \
  --freeze_backbone \
  --device cuda:0 --out_dir checkpoints/sft

# === Stage 2 GRPO (烧 worker API) ==================================
# 注意：mode flags 必须和 SFT 阶段一致 (否则 ckpt 加载会缺/多 key)
python scripts/06_run_grpo.py \
  --head_ckpt checkpoints/sft/head_epoch4 \
  --lora_rank 32 \
  --dataset gsm8k --n 200 --G 4 --epochs 2 \
  --worker openai --worker_model deepseek-chat \
  --out_dir checkpoints/grpo

# === Eval ==========================================================
python scripts/05_evaluate.py \
  --mode head --head_ckpt checkpoints/grpo/head_grpo_final \
  --lora_rank 32 \
  --dataset gsm8k --n 200 \
  --worker openai --worker_model deepseek-chat \
  --out_jsonl results/apo_gsm8k.jsonl

# === Inspect head outputs on a few sample tasks =====================
python scripts/04_inspect_head.py --ckpt checkpoints/sft/head_epoch4
```

## Layout

```
src/arch_policy/
  config.py                 ArchSpec / ModelSpec / TrainSpec
  architecture/
    spec.py                 ArchLogits, ArchTargets typed dataclasses
    sampler.py              sample_arch + Plackett-Luce + 4 typed log_prob
    library.py              93 NamedArch (68 canonical / 15 imperfect / 10 random,
                            33 canonical families, 8 roles)
    encoder.py              NamedArch → ArchTargets
  head/model.py             latent agent embedding + 4 typed heads (gate/role/edge/seq)
  executor/
    prompts.py              8 role prompts + ReAct + Synth
    tools.py                python_exec / sympy_check / web_search
    agent.py                ReAct inner loop (over `step`s)
    synth.py                ANSWER:/CONTINUE judge
    multi_agent.py          main exec loop (cycle → turn → step)
    openai_worker.py        DeepSeek / OpenAI API worker
  data/                     6 benchmark loaders (GSM8K / MATH / HumanEval / MBPP /
                            MMLU / BBH / ARC) + SFT dataset (family-stratified)
  reward/                   composite reward + family-aware grader
  training/
    sft.py                  4 typed losses (BCE+CE+BCE+PL-NLL, label smoothing 0.05)
    grpo.py                 typed log_pi + entropy bonus, no KL
  baselines.py              11 fixed-topology baselines

scripts/
  00_setup_env.sh           conda env
  01_download_models.py     pull Qwen3-4B (override via HEAD_MODEL env)
  02_smoke_test.py          26 CPU smoke tests
  03_run_sft.py             Stage-1 (--stratify_by_family / --tier_ratio)
  04_inspect_head.py        decode head outputs
  05_evaluate.py            baseline / head eval
  06_run_grpo.py            Stage-2

tests/                      26 unit tests
configs/default.yaml        config snapshot (kept in sync with config.py dataclasses)
```
