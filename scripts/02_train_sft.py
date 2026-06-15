"""End-to-end Stage-1 SFT runner.

Usage::

    # Quick CPU smoke (synthetic tasks only)
    python scripts/02_train_sft.py --tasks_source synthetic --n_tasks 30 \
        --epochs 1 --batch_size 4 --max_steps 10 --device cpu --dtype float32 \
        --out_dir /tmp/sft_smoke

    # Mixed 11-source SFT (default ~11.5K task pool — see DEFAULT_SFT_MIX)
    python scripts/02_train_sft.py --tasks_source mixed --epochs 3 \
        --batch_size 8 --device cuda:0 --out_dir checkpoints/sft

    # Single-source (e.g. GSM8K only)
    python scripts/02_train_sft.py --tasks_source gsm8k --n_tasks 1500 --epochs 3
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
    bench,
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
    src = ap.add_mutually_exclusive_group()
    src.add_argument(
        "--tasks_source",
        choices=["synthetic", "gsm8k", "humaneval", "mbpp", "math",
                 "mmlu", "bbh", "arc", "mixed"],
        default=None,
        help="Legacy single-source / mixed loader. `mixed` uses the "
             "11-source DEFAULT_SFT_MIX (~11.5K tasks).",
    )
    src.add_argument(
        "--bench", choices=bench.available(), default=None,
        help="Per-bench plugin (one file per bench). Picks loader + "
             "SFT pool from `bench/<name>.py`. Mutually exclusive with "
             "--tasks_source.",
    )
    ap.add_argument("--n_tasks", type=int, default=600,
                    help="Per-source cap (ignored for `mixed` which uses DEFAULT_SFT_MIX).")
    ap.add_argument("--worker_models", default=None,
                    help="Comma-separated model pool. When set (>1 model) the "
                         "head gets the 5th typed dim (head_M) and SFT trains "
                         "it against RANDOM per-slot model targets → uniform/"
                         "unbiased model prior. GRPO then learns task→model.")
    ap.add_argument("--sft_pool", choices=["bench", "full"], default="bench",
                    help="SFT architectural prior. `bench`: the per-bench "
                         "curated pool (pool_ratio forced 1.0). `full`: the "
                         "comprehensive bench-agnostic `full_library` with the "
                         "default 2-tier sampler — recommended so one broad "
                         "prior serves every benchmark.")
    ap.add_argument("--epochs", type=int, default=TRAIN.sft_epochs)
    ap.add_argument("--batch_size", type=int, default=TRAIN.sft_batch_size)
    ap.add_argument("--grad_accum", type=int, default=TRAIN.sft_grad_accum,
                    help="Gradient accumulation steps. effective_batch = "
                         "batch_size * grad_accum. Used when GPU OOMs at "
                         "the per-step batch but you want a bigger effective "
                         "batch (e.g. SFT @ seq=2048 needs batch=2 + accum=8 "
                         "to fit 80GB H100 without flash-attn).")
    ap.add_argument("--lr", type=float, default=None,
                    help="Learning rate. If unset, picked automatically by mode: "
                         "5e-5 (frozen backbone) / 1e-4 (LoRA) / 1e-5 (full FT).")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--save_every", type=int, default=TRAIN.sft_save_every_n_steps)
    ap.add_argument("--head_model", default=MODEL.head_model)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="checkpoints/sft")
    ap.add_argument("--seed", type=int, default=42,
                    help="production seed (match 03/04 so the bench "
                         "train/test split is consistent across stages).")
    ap.add_argument("--max_seq_len", type=int, default=1024,
                    help="Max tokens for the head's tokenizer "
                         "(TrainSpec.tokenizer_max_len). 1024 covers "
                         "96.6%% of HLE tasks; raise if your benchmark "
                         "has longer prompts.")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])

    # ---- Trainability mode ------------------------------------------------
    ap.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="If set, only the typed heads are trained (heads-only "
                         "baseline). Default OFF: backbone is also trainable.")
    ap.add_argument("--lora_rank", type=int, default=32,
                    help="LoRA rank on backbone (q/k/v/o + MLP projections). "
                         "Default 32. Set 0 to disable LoRA (full FT).")
    ap.add_argument("--lora_alpha", type=int, default=64,
                    help="LoRA scaling alpha (only used if --lora_rank > 0).")
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Trade compute for activation memory (production: "
                         "ON — fits LoRA-32 seq=1024 on one 80GB card with "
                         "headroom). Pass --no-gradient_checkpointing to "
                         "disable.")
    ap.add_argument("--pool_ratio", type=float, default=0.85,
                    help="2-tier sampler: P(draw from SFT pool [82 canonical "
                         "+ 15 imperfect = 97 hand-designed archs, uniform "
                         "internal]). 1 - pool_ratio goes to TRUE on-demand "
                         "random (fresh valid arch per draw, reproducible via "
                         "per-(epoch,task) seed). Default 0.85 → 85%% pool / "
                         "15%% true random.")
    ap.add_argument("--no_random_on_demand", action="store_true",
                    help="Use the legacy 10 pre-generated random archs "
                         "(reused every epoch) instead of true on-demand "
                         "random. Off by default — on-demand is the recommended "
                         "regulariser.")
    ap.add_argument("--legacy_tier_ratio", type=float, nargs=3, default=None,
                    metavar=("CANONICAL", "IMPERFECT", "RANDOM"),
                    help="ABLATION ONLY: revert to the old 3-tier "
                         "family-stratified sampler (e.g. 0.75 0.15 0.10). "
                         "Overrides --pool_ratio when set.")
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

    if args.bench is not None:
        adapter = bench.get(args.bench)
        tasks = adapter.load_split("train", seed=args.seed)
        source_label = f"bench={args.bench}"
    else:
        src = args.tasks_source or "mixed"
        if src == "synthetic":
            tasks = load_local_synthetic(
                n_per_family=max(1, args.n_tasks // 3), seed=args.seed,
            )
        elif src == "mixed":
            tasks = load_mixed(seed=args.seed)
        else:
            tasks = load_huggingface(src, split="train",
                                    n=args.n_tasks, seed=args.seed)
        source_label = src
    print(f"[sft] loaded {len(tasks)} tasks from {source_label}")

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

    use_full_lib = (args.sft_pool == "full") or (args.bench is None)
    library = (full_library(seed=args.seed) if use_full_lib
               else bench.get(args.bench).get_pool())
    lib_label = "full_library" if use_full_lib else f"{args.bench}_pool"
    print(f"[sft] library = {lib_label} (size {len(library)})")
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
    from dataclasses import replace as _replace
    model_names = (tuple(m.strip() for m in args.worker_models.split(",") if m.strip())
                   if args.worker_models else ARCH.model_names)
    arch_spec = _replace(ARCH, model_names=model_names)
    print(f"[sft] n_models={arch_spec.n_models}"
          + (f" (pool: {', '.join(model_names)}) — head_M trained toward "
             f"UNIFORM via deterministic max-entropy loss" if arch_spec.n_models > 1
             else " (single-model; no head_M)"))
    model = ArchitectureHead(
        backbone_name=args.head_model,
        arch_spec=arch_spec,
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

    # A CURATED per-bench pool IS the prior (pool_ratio 1.0, no random tier).
    # The comprehensive full_library uses the default 2-tier sampler.
    legacy_tier = (tuple(args.legacy_tier_ratio)
                   if args.legacy_tier_ratio is not None else None)
    eff_pool_ratio = args.pool_ratio if use_full_lib else 1.0
    eff_random_on_demand = ((not args.no_random_on_demand) if use_full_lib
                            else False)
    ds = SFTArchDataset(
        tasks=tasks, library=library, targets=targets,
        tokenizer=tokenizer, max_len=args.max_seq_len, seed=args.seed,
        pool_ratio=eff_pool_ratio,
        random_on_demand=eff_random_on_demand,
        legacy_tier_ratio=legacy_tier,
    )
    # Log the EFFECTIVE config, not the raw CLI flags (the --bench
    # path overrides pool_ratio and random_on_demand).
    if legacy_tier is not None:
        print(
            f"[sft] LEGACY 3-tier family-stratified sampler (ablation): "
            f"canonical/imperfect/random = "
            f"{legacy_tier[0]:.2f}/{legacy_tier[1]:.2f}/{legacy_tier[2]:.2f}"
        )
    elif not use_full_lib:
        print(
            f"[sft] bench={args.bench} sampler: "
            f"pool_ratio=1.0 (forced) over {len(library)} archs, "
            f"random_tier=DISABLED — curated bench pool IS the prior"
        )
    else:
        rand_mode = ("on-demand (true random per draw)"
                     if eff_random_on_demand
                     else "FIXED 10 pre-generated archs (legacy)")
        print(
            f"[sft] 2-tier sampler: pool_ratio={eff_pool_ratio:.2f} "
            f"over {len(library)} archs; "
            f"random tier ({1-eff_pool_ratio:.2f}) = {rand_mode}"
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
        sft_grad_accum=args.grad_accum,
        sft_save_every_n_steps=args.save_every,
        sft_max_steps=args.max_steps,
        tokenizer_max_len=args.max_seq_len,
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[sft] starting training: epochs={args.epochs}, batch={args.batch_size}, lr={lr:g}")
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
