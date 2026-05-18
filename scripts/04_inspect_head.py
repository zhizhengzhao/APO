"""Inspect a trained head: print typed distributions + sampled architectures.

Usage::

    python scripts/04_inspect_head.py --ckpt checkpoints/sft/head_step100
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

from arch_policy import (
    ARCH,
    ArchitectureHead,
    load_local_synthetic,
    load_tokenizer,
    sample_arch,
    seed_all,
    to_arch_logits,
)
from arch_policy.training.sft import load_head_checkpoint


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--head_model", default="Qwen/Qwen3-4B")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_tasks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora_rank", type=int, default=0,
                    help="Match SFT setting if non-zero (else PEFT keys won't load).")
    ap.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--dtype", default="float32",
                    help="bfloat16 / float32 — float32 safe for inspection on CPU.")
    args = ap.parse_args()

    seed_all(args.seed)

    tasks = load_local_synthetic(n_per_family=2, seed=args.seed)[: args.n_tasks]
    tok = load_tokenizer(args.head_model)
    model = ArchitectureHead(
        backbone_name=args.head_model,
        freeze_backbone=args.freeze_backbone,
        lora_rank=args.lora_rank,
        torch_dtype=args.dtype,
    )
    print(f"[inspect] loading checkpoint {args.ckpt}")
    load_head_checkpoint(model, args.ckpt)
    model.to(args.device).eval()

    enc = tok([t.task for t in tasks], padding=True, truncation=True, return_tensors="pt").to(args.device)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

    print(f"\n=== Head global stats over {len(tasks)} tasks ===")
    g_p = torch.sigmoid(out["gate_logits"]).cpu()
    print(f"  gate_prob:   mean={g_p.mean():.3f}  range=[{g_p.min():.3f}, {g_p.max():.3f}]")
    Q = F.softmax(out["role_logits"], dim=-1).cpu()
    print(f"  role_prob:   max-along-R mean={Q.max(-1).values.mean():.3f}")
    e_p = torch.sigmoid(out["edge_logits"]).cpu()
    print(f"  edge_prob:   mean={e_p.mean():.3f}  range=[{e_p.min():.3f}, {e_p.max():.3f}]")
    s = out["seq_scores"].cpu()
    print(f"  seq_scores:  abs_mean={s.abs().mean():.3f}")

    for i, t in enumerate(tasks):
        logits = to_arch_logits(out, batch_idx=i)
        # CPU-side determinstic decode for inspection
        logits_cpu = type(logits)(
            gate_logits=logits.gate_logits.cpu(),
            role_logits=logits.role_logits.cpu(),
            edge_logits=logits.edge_logits.cpu(),
            seq_scores=logits.seq_scores.cpu(),
        )
        arch = sample_arch(logits_cpu, deterministic=True)

        print("\n----")
        print(f"task #{i} ({t.family}): {t.task[:80]}{'...' if len(t.task) > 80 else ''}")
        print(f"  gate prob: {[f'{p:.2f}' for p in torch.sigmoid(logits_cpu.gate_logits).tolist()]}")
        print(f"  argmax roles: ", end="")
        argmax_roles = logits_cpu.role_logits.argmax(-1).tolist()
        print([ARCH.role_names[r] for r in argmax_roles])
        print(f"  arch: {arch.to_summary()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
