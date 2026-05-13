"""Stage-2: architecture-level GRPO fine-tuning (v3 typed log_pi, no KL).

Loads a SFT-warm-started head, samples G architectures per task, executes
them via the chosen worker, and updates the head with group-relative
advantage policy gradient over the typed distributions.

Usage example:

    python scripts/06_run_grpo.py \
        --head_ckpt checkpoints/sft/head_step100 \
        --dataset gsm8k --n 200 \
        --worker openai --worker_model deepseek-chat \
        --G 4 --epochs 2 \
        --out_dir checkpoints/grpo_v1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch


def _build_worker(args):
    if args.worker == "mock":
        from arch_policy import MockWorker
        return MockWorker(fake_answer=args.mock_answer)
    if args.worker == "openai":
        from arch_policy import OpenAIWorker
        return OpenAIWorker(
            model=args.worker_model,
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
            temperature=args.worker_temperature,
            timeout=args.worker_timeout,
        )
    if args.worker == "hf":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from arch_policy import HFWorker
        tok = AutoTokenizer.from_pretrained(args.worker_model)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(
            args.worker_model, torch_dtype="bfloat16", device_map="auto",
        )
        return HFWorker(model=mdl, tokenizer=tok)
    raise ValueError(args.worker)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head_ckpt", required=True)
    ap.add_argument("--head_model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--head_dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--head_device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--dataset", default="gsm8k", choices=["synthetic", "gsm8k", "humaneval", "math"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=200, help="number of training tasks")

    ap.add_argument("--worker", choices=["mock", "openai", "hf"], default="openai")
    ap.add_argument("--worker_model", default="deepseek-chat")
    ap.add_argument("--worker_temperature", type=float, default=0.0)
    ap.add_argument("--worker_timeout", type=float, default=120.0)
    ap.add_argument("--mock_answer", default="42")

    ap.add_argument("--G", type=int, default=4, help="GRPO group size")
    ap.add_argument("--batch_size", type=int, default=4, help="tasks per GRPO step")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--entropy_weight", type=float, default=0.01)
    ap.add_argument("--max_seq_len", type=int, default=384)
    ap.add_argument("--max_new_tokens", type=int, default=1024)

    ap.add_argument("--out_dir", default="checkpoints/grpo")
    ap.add_argument("--save_every", type=int, default=20)
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default="arch_policy")
    ap.add_argument("--wandb_run", default=None)

    args = ap.parse_args()

    from arch_policy import (
        ARCH, ArchitectureHead, GRPOBatch, MultiAgentExecutor,
        TRAIN, load_head_checkpoint, load_tokenizer, seed_all, train_grpo,
    )
    from arch_policy.data.tasks import load_huggingface, load_local_synthetic

    seed_all(args.seed)

    # --- Load tasks ---
    if args.dataset == "synthetic":
        tasks = load_local_synthetic(n_per_family=max(1, args.n // 3), seed=args.seed)[: args.n]
    else:
        tasks = load_huggingface(args.dataset, split=args.split, n=args.n, seed=args.seed)
    print(f"[grpo] loaded {len(tasks)} tasks ({args.dataset}/{args.split})")

    # --- Worker + executor ---
    worker = _build_worker(args)
    executor = MultiAgentExecutor(worker=worker, max_new_tokens_per_call=args.max_new_tokens)

    # --- Head ---
    tokenizer = load_tokenizer(args.head_model)
    model = ArchitectureHead(
        backbone_name=args.head_model,
        freeze_backbone=True,
        torch_dtype=args.head_dtype if args.head_device.startswith("cuda") else "float32",
    )
    load_head_checkpoint(model, args.head_ckpt)
    model.to(args.head_device)
    model.train()
    print(f"[grpo] head loaded from {args.head_ckpt}; trainable params = {model.trainable_parameters():,}")

    # --- TrainSpec override ---
    spec = replace(
        TRAIN,
        grpo_group_size=args.G,
        grpo_lr=args.lr,
        grpo_batch_size=args.batch_size,
        grpo_entropy_weight=args.entropy_weight,
    )

    # --- Wandb ---
    wb = None
    if args.wandb:
        try:
            import wandb
            wb = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run or f"grpo_{args.dataset}_{int(time.time())}",
                config=vars(args),
            )
        except ImportError:
            print("[grpo] wandb not installed; skipping logging")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Build batches ---
    # Group tasks into mini-batches of size args.batch_size; iterate args.epochs times.
    batches = []
    for epoch in range(args.epochs):
        # naive shuffle: rotate the list
        for i in range(0, len(tasks), args.batch_size):
            chunk = tasks[i : i + args.batch_size]
            if not chunk:
                continue
            batches.append(GRPOBatch(
                task_texts=[t.task for t in chunk],
                gold_answers=[t.gold_answer for t in chunk],
                task_samples=list(chunk),
            ))
    print(f"[grpo] total {len(batches)} GRPO steps (G={args.G})")

    info = train_grpo(
        model=model,
        tokenizer=tokenizer,
        batches=batches,
        executor=executor,
        spec=spec,
        out_dir=out_dir,
        device=args.head_device,
        log_every=args.log_every,
        save_every=args.save_every,
        wandb_run=wb,
    )
    if wb is not None:
        wb.finish()
    print(f"[grpo] DONE — checkpoints under {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
