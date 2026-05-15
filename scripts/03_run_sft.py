"""End-to-end Stage-1 SFT runner.

Usage::

    # Quick CPU smoke (synthetic tasks only)
    python scripts/03_run_sft.py --tasks_source synthetic --n_tasks 30 \
        --epochs 1 --batch_size 4 --max_steps 10 --device cpu --dtype float32 \
        --out_dir /tmp/sft_smoke

    # Mixed 6-source SFT (default 5000 task pool)
    python scripts/03_run_sft.py --tasks_source mixed --epochs 3 \
        --batch_size 8 --device cuda:0 --out_dir checkpoints/sft

    # Single-source (e.g. GSM8K only)
    python scripts/03_run_sft.py --tasks_source gsm8k --n_tasks 1500 --epochs 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from arch_policy import (
    ARCH,
    MODEL,
    TRAIN,
    ArchitectureHead,
    SFTArchDataset,
    encode_library,
    full_library,
    load_local_synthetic,
    load_mixed,
    load_tokenizer,
    seed_all,
    train_sft,
)
from arch_policy.data.tasks import load_huggingface


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tasks_source",
        choices=["synthetic", "gsm8k", "humaneval", "mbpp", "math",
                 "mmlu", "bbh", "arc", "mixed"],
        default="mixed",
        help="`mixed` uses the 6-source DEFAULT_SFT_MIX (~5000 tasks).",
    )
    ap.add_argument("--n_tasks", type=int, default=600,
                    help="Per-source cap (ignored for `mixed` which uses DEFAULT_SFT_MIX).")
    ap.add_argument("--epochs", type=int, default=TRAIN.sft_epochs)
    ap.add_argument("--batch_size", type=int, default=TRAIN.sft_batch_size)
    ap.add_argument("--lr", type=float, default=None,
                    help="Learning rate. If unset, picked automatically by mode: "
                         "5e-5 (frozen backbone) / 1e-4 (LoRA) / 1e-5 (full FT).")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--save_every", type=int, default=TRAIN.sft_save_every_n_steps)
    ap.add_argument("--head_model", default=MODEL.head_model)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="checkpoints/sft")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_seq_len", type=int, default=512,
                    help="Max tokens for the head's tokenizer (TrainSpec.tokenizer_max_len)")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])

    # ---- Trainability mode (V3.5) ------------------------------------------
    ap.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="If set, only the typed heads are trained (V3 behavior). "
                         "Default OFF in V3.5: backbone is also trainable.")
    ap.add_argument("--lora_rank", type=int, default=0,
                    help="If > 0, wrap the backbone with PEFT LoRA at this rank "
                         "(recommended for 9B+ on memory-constrained GPUs). "
                         "Default 0 = no LoRA.")
    ap.add_argument("--lora_alpha", type=int, default=64,
                    help="LoRA scaling alpha (only used if --lora_rank > 0).")
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Trade compute for activation memory. ESSENTIAL for "
                         "full FT 9B on a single 80GB card.")
    ap.add_argument("--stratify_by_family", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Pair each task with a family-stratified random NamedArch "
                         "(uniform over families, not entries). Default ON. "
                         "Pass --no-stratify_by_family to revert to uniform-over-entries.")
    ap.add_argument("--tier_ratio", type=float, nargs=3,
                    default=[0.73, 0.16, 0.11],
                    metavar=("CANONICAL", "IMPERFECT", "RANDOM"),
                    help="Tier sampling ratio when stratify_by_family is on. "
                         "Default (0.73, 0.16, 0.11) matches library composition.")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default="arch_policy")
    ap.add_argument("--wandb_run", default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    seed_all(args.seed)

    print(f"[sft] device={args.device} head_model={args.head_model}")
    print(f"[sft] arch_spec n_max={ARCH.n_max} roles={ARCH.k_roles} d_latent={ARCH.d_latent}")
    print(f"[sft] roles: {ARCH.role_names}")

    if args.tasks_source == "synthetic":
        tasks = load_local_synthetic(n_per_family=max(1, args.n_tasks // 3), seed=args.seed)
    elif args.tasks_source == "mixed":
        tasks = load_mixed(seed=args.seed)
    else:
        tasks = load_huggingface(args.tasks_source, split="train", n=args.n_tasks, seed=args.seed)
    print(f"[sft] loaded {len(tasks)} tasks from {args.tasks_source}")

    wb = None
    if args.wandb:
        try:
            import wandb
            wb = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run or f"sft_{args.tasks_source}",
                config=vars(args),
            )
            print(f"[sft] wandb initialized: {wb.url}")
        except ImportError:
            print("[sft] wandb not installed, skipping logging")

    library = full_library(seed=args.seed)
    print(f"[sft] library size = {len(library)}")
    targets = encode_library(library)
    print(f"[sft] encoded {len(targets)} typed targets")

    print(f"[sft] loading tokenizer from {args.head_model} ...")
    tokenizer = load_tokenizer(args.head_model)
    print(f"[sft] vocab_size={len(tokenizer)}")

    print(f"[sft] loading head with backbone {args.head_model} (also downloads if needed) ...")
    if args.lora_rank > 0:
        mode_str = f"LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})"
    elif args.freeze_backbone:
        mode_str = "frozen backbone (heads only)"
    else:
        mode_str = "full fine-tune"
    print(f"[sft] trainability mode: {mode_str}; gradient_checkpointing={args.gradient_checkpointing}")
    model = ArchitectureHead(
        backbone_name=args.head_model,
        arch_spec=ARCH,
        freeze_backbone=args.freeze_backbone,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_checkpointing=args.gradient_checkpointing,
        torch_dtype=args.dtype if args.device.startswith("cuda") else "float32",
    )
    n_trainable = model.trainable_parameters()
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[sft] trainable params = {n_trainable:,} / {n_total:,} "
          f"({100 * n_trainable / max(1, n_total):.2f}%)")

    tier_ratio_t = tuple(args.tier_ratio)
    ds = SFTArchDataset(
        tasks=tasks,
        library=library,
        targets=targets,
        tokenizer=tokenizer,
        max_len=args.max_seq_len,
        seed=args.seed,
        stratify_by_family=args.stratify_by_family,
        tier_ratio=tier_ratio_t,
    )
    if args.stratify_by_family:
        print(
            f"[sft] family-stratified sampling enabled (recommended). "
            f"tier_ratio canonical/imperfect/random = "
            f"{tier_ratio_t[0]:.2f}/{tier_ratio_t[1]:.2f}/{tier_ratio_t[2]:.2f}"
        )
    else:
        print(
            "[sft] family-stratified sampling DISABLED — uniform over entries; "
            "high-variant families (mad_debate / moa_fanin / hier / hub) will be "
            "OVERSAMPLED. Pass --stratify_by_family to fix."
        )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=ds.collate,
        num_workers=0,
    )

    if args.lr is not None:
        lr = args.lr
    elif args.lora_rank > 0:
        lr = 1e-4
    elif args.freeze_backbone:
        lr = 5e-5
    else:
        lr = 1e-5
    print(f"[sft] LR = {lr:g}")

    from dataclasses import replace
    spec = replace(
        TRAIN,
        sft_epochs=args.epochs,
        sft_lr=lr,
        sft_batch_size=args.batch_size,
        sft_save_every_n_steps=args.save_every,
        sft_max_steps=args.max_steps,
        tokenizer_max_len=args.max_seq_len,
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[sft] starting training: epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")
    info = train_sft(
        model=model,
        train_loader=loader,
        spec=spec,
        out_dir=out,
        log_every=5,
        device=args.device,
        wandb_run=wb,
    )
    print(f"[sft] DONE. final_step={info['final_step']}")
    print(f"[sft] checkpoints under: {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
