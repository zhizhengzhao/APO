#!/usr/bin/env bash
# Launch one task-category GRPO run from the shared SFT warm-start.
#
# Usage:
#   run_cat_rl.sh <cat> <gpu> <max_new_tokens> <max_seq_len> [epochs]
#   e.g.  run_cat_rl.sh cat_code 0 8192 4096 18
#
# Paths are taken from the environment (no hardcoded machine paths):
#   SFT_CKPT  : shared SFT warm-start head dir       (default: checkpoints/sft_shared/head_final)
#   OUT_ROOT  : where per-category run dirs are written (default: runs)
#
# Hyperparameters match the released runs: G=4, batch=16, lr=3e-5, 18 epochs,
# Qwen 3-tier pool (flash/plus/max, thinking OFF), cost_bonus_scale=0.5.
set -eo pipefail

CAT=${1:?usage: run_cat_rl.sh <cat> <gpu> <mnt> <seq> [epochs]}
GPU=${2:?need GPU id}
MNT=${3:-8192}
SEQ=${4:-4096}
EPOCHS=${5:-18}

SFT_CKPT=${SFT_CKPT:-checkpoints/sft_shared/head_final}
OUT_ROOT=${OUT_ROOT:-runs}

export PYTHONPATH=src
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPU
# Math leans on heavy symbolic compute (sympy); give python_exec more wall time.
if [ "$CAT" = "cat_math" ]; then export ARCH_PYTHON_TIMEOUT_S=120; fi

OUT="$OUT_ROOT/${CAT}/grpo_${CAT}_v2"
mkdir -p "$OUT"

exec python3 scripts/03_train_grpo.py \
  --bench "$CAT" --n 0 --train_ratio 1.0 \
  --head_ckpt "$SFT_CKPT" \
  --lora_rank 32 --gradient_checkpointing \
  --worker qwen --worker_model qwen3.7-max \
  --worker_models qwen3.6-flash,qwen3.6-plus,qwen3.7-max \
  --worker_temperature 0.6 --worker_timeout 600 \
  --judge qwen --judge_model qwen3.7-max --judge_timeout 180 \
  --batch_size 16 --G 4 --epochs "$EPOCHS" --lr 3e-5 --cost_bonus_scale 0.5 \
  --max_seq_len "$SEQ" --max_new_tokens "$MNT" \
  --max_concurrent_runs 64 --safety_max_cycles 8 --safety_max_steps 16 \
  --wall_clock_timeout_s 400 --max_llm_calls_per_trace 32 \
  --cache_path "$OUT/arch_cache.jsonl" --cache_reuse_prob 0.0 \
  --inject_mode none --save_every 25 --log_every 1 \
  --out_dir "$OUT" --seed 42
