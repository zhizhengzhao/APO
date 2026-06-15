"""SFT trainer for the architecture head — 4 typed losses, label smoothing.

  L = w_g·L_gate + w_Q·L_role + w_e·L_edge + w_s·L_seq

  L_gate  = BCE over all N slots, target smoothed 0/1 → 0.05/0.95
  L_role  = CE on active slots only, smoothed onehot
  L_edge  = BCE on active (i,j) pairs (no diag), smoothed
  L_seq   = Plackett-Luce NLL on teacher perm (NOT smoothed; listwise rank
            + stochastic sampling already provide diversity)

Smoothing 0.05 prevents collapse onto sharp NamedArch attractors, leaving
margin for GRPO to sample interpolations. No KL — the head is a policy
planner, not a language model that needs language-prior protection.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..architecture.sampler import log_prob_pl
from ..architecture.spec import ArchLogits, ArchTargets, active_pair_mask
from ..config import TRAIN, TrainSpec
from ..head.model import ArchitectureHead


# ---------------------------------------------------------------------------
# Per-architecture typed loss
# ---------------------------------------------------------------------------

def _smooth_bernoulli(target: torch.Tensor, eps: float) -> torch.Tensor:
    """0/1 → eps/(1-eps) for label smoothing."""
    return target.float() * (1 - 2 * eps) + eps  # 0 → eps, 1 → 1 - eps


def _smoothed_ce_from_onehot(
    logits: torch.Tensor,
    target_idx: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Cross-entropy with label smoothing (onehot → (1-eps)·oh + eps/R).

    `logits` is [..., R] and `target_idx` is [...] long indices.
    Thin wrapper around F.cross_entropy(..., label_smoothing=eps) for clarity.
    """
    return F.cross_entropy(
        logits, target_idx, label_smoothing=eps, reduction="mean",
    )


def sft_loss_single(
    logits: ArchLogits,
    target: ArchTargets,
    spec: TrainSpec | None = None,
) -> dict[str, torch.Tensor]:
    """Compute typed losses for one architecture, with label smoothing.

    Returns a dict with components and `total`. All components are
    *averaged* per-element so they're roughly comparable in scale.

    Smoothing only applied to BCE (gate/edge) and CE (role); PL (seq) is left
    as a hard NLL since its listwise structure already provides diversity.
    """
    if spec is None:
        spec = TRAIN

    eps = spec.sft_label_smoothing
    active = target.gates.bool()

    # ---- gate: BCE over all N slots, with label smoothing ------------------
    smoothed_gates = _smooth_bernoulli(target.gates, eps)
    gate_loss = F.binary_cross_entropy_with_logits(
        logits.gate_logits, smoothed_gates, reduction="mean",
    )

    # ---- role: CE over active slots only, with label smoothing -------------
    if active.any():
        role_loss = _smoothed_ce_from_onehot(
            logits.role_logits[active],
            target.roles[active],
            eps=eps,
        )
    else:
        role_loss = torch.zeros((), device=logits.gate_logits.device)

    # ---- edge: BCE over active (i,j) pairs (no diag), with smoothing -------
    pair = active_pair_mask(active)
    if pair.any():
        smoothed_edges = _smooth_bernoulli(target.edges[pair], eps)
        edge_loss = F.binary_cross_entropy_with_logits(
            logits.edge_logits[pair],
            smoothed_edges,
            reduction="mean",
        )
    else:
        edge_loss = torch.zeros((), device=logits.gate_logits.device)

    # ---- seq: Plackett-Luce NLL, NO smoothing ------------------------------
    if target.seq.numel() > 0:
        seq_loss = -log_prob_pl(logits.seq_scores, target.seq) / target.seq.numel()
    else:
        seq_loss = torch.zeros((), device=logits.gate_logits.device)

    # ---- model: push head_M toward UNIFORM (deterministic max-entropy) -----
    # Multi-model only (model_logits present iff n_models > 1). There is no
    # task→model grounding in the SFT pool (same as the topology dims), so the
    # SFT target for the 5th dim is a uniform per-agent model distribution. We
    # use the DETERMINISTIC cross-entropy-to-uniform = -mean(log_softmax) =
    # KL(uniform‖p) + const (optimum p=uniform, value ln(n_models)). Same
    # EXPECTED gradient as sampling random one-hot targets, but ZERO variance
    # + no RNG dependence → reproducible SFT + deterministic eval. GRPO then
    # learns the real task→model preferences from reward. Active slots only.
    if logits.model_logits is not None and active.any():
        logp = F.log_softmax(logits.model_logits[active], dim=-1)
        model_loss = -logp.mean()
    else:
        model_loss = torch.zeros((), device=logits.gate_logits.device)

    total = (
        spec.sft_w_gate * gate_loss
        + spec.sft_w_role * role_loss
        + spec.sft_w_edge * edge_loss
        + spec.sft_w_seq  * seq_loss
        + spec.sft_w_model * model_loss
    )
    return {
        "gate":  gate_loss,
        "role":  role_loss,
        "edge":  edge_loss,
        "seq":   seq_loss,
        "model": model_loss,
        "total": total,
    }


def sft_loss_batch(
    head_out: dict[str, torch.Tensor],
    targets: Sequence[ArchTargets],
    spec: TrainSpec | None = None,
) -> dict[str, torch.Tensor]:
    """Mean of `sft_loss_single` over the batch (variable-K is per-sample).

    `head_out` is the dict returned by `ArchitectureHead.forward`, with a
    leading [B, ...] dim.
    """
    B = head_out["gate_logits"].shape[0]
    if len(targets) != B:
        raise ValueError(f"len(targets)={len(targets)} != batch size {B}")

    accum = {"gate": [], "role": [], "edge": [], "seq": [], "model": [], "total": []}
    model_out = head_out.get("model_logits")
    for b in range(B):
        logits = ArchLogits(
            gate_logits=head_out["gate_logits"][b],
            role_logits=head_out["role_logits"][b],
            edge_logits=head_out["edge_logits"][b],
            seq_scores =head_out["seq_scores"][b],
            model_logits=(model_out[b] if model_out is not None else None),
        )
        # move target to head device
        device = head_out["gate_logits"].device
        target = ArchTargets(
            gates=targets[b].gates.to(device),
            roles=targets[b].roles.to(device),
            edges=targets[b].edges.to(device),
            seq=targets[b].seq.to(device),
        )
        comp = sft_loss_single(logits, target, spec)
        for k, v in comp.items():
            accum[k].append(v)

    return {k: torch.stack(v).mean() for k, v in accum.items()}


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _ensure_dir(p: str | os.PathLike) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_head_checkpoint(model: ArchitectureHead, ckpt_dir: str | os.PathLike, tag: str) -> Path:
    """Save only the trainable parameters.

    Works across all three trainability modes (frozen head-only / full FT /
    LoRA). Sizes vary accordingly: ~1MB heads-only, ~8GB full-FT 4B,
    ~50MB LoRA rank-32. We use `requires_grad` to decide what to save so
    the checkpoint always matches what was actually trained.
    """
    out = _ensure_dir(ckpt_dir) / f"head_{tag}"
    out.mkdir(parents=True, exist_ok=True)

    trainable_keys = {n for n, p in model.named_parameters() if p.requires_grad}
    full_state = model.state_dict()
    saved = {k: v.detach().cpu() for k, v in full_state.items() if k in trainable_keys}

    state = {
        "head_state_dict": saved,
        "head_cfg": asdict(model.head_cfg),
        "arch_spec": asdict(model.arch_spec),
        "backbone_name": model.backbone_name,
        "saved_keys": sorted(saved.keys()),
    }
    torch.save(state, out / "head.pt")
    with open(out / "meta.json", "w") as f:
        json.dump({
            "backbone_name": model.backbone_name,
            "arch_spec": asdict(model.arch_spec),
            "head_cfg": asdict(model.head_cfg),
            "n_saved_keys": len(saved),
            "n_saved_params": sum(v.numel() for v in saved.values()),
        }, f, indent=2)
    return out


def load_head_checkpoint(model: ArchitectureHead, ckpt_dir: str | os.PathLike) -> None:
    """Load a checkpoint into `model`.

    Uses strict=False so missing backbone keys are silently OK — the
    freshly-loaded backbone weights stay intact when the checkpoint only
    saved trainable params (LoRA / heads-only mode).

    Refuses to load if any TYPED HEAD key (head_g/head_Q/head_S/M/B/b0
    — what actually outputs architecture decisions) is missing from the
    ckpt. Silent random-init of a typed head looks like a successful
    load but resets all trained knowledge for that channel.
    """
    state = torch.load(Path(ckpt_dir) / "head.pt", map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state["head_state_dict"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys when loading checkpoint: {unexpected}")
    # `missing` may include backbone params if the checkpoint was made with
    # frozen backbone and we now load with unfrozen — that's intentional.
    # But ANY typed-head key missing means we silently lost that head's
    # learned weights. Surface loudly.
    TYPED_HEAD_PREFIXES = ("head_g.", "head_Q.", "head_S.", "head_M.",
                            "agent_proj.", "slot_emb.", "body.")
    TYPED_HEAD_PARAMS = ("M", "B", "b0")   # bare module-level params
    typed_missing = [k for k in missing
                     if k in TYPED_HEAD_PARAMS
                     or any(k.startswith(p) for p in TYPED_HEAD_PREFIXES)]
    if typed_missing:
        raise RuntimeError(
            f"load_head_checkpoint: typed-head keys MISSING from ckpt "
            f"{ckpt_dir} — would silently leave them at random init. "
            f"Refusing to proceed.\n  missing typed keys: {typed_missing}"
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_sft(
    model: ArchitectureHead,
    train_loader: DataLoader,
    spec: TrainSpec | None = None,
    out_dir: str | os.PathLike = "checkpoints/sft",
    log_every: int = 5,
    eval_loader: DataLoader | None = None,
    device: str | None = None,
    wandb_run=None,
) -> dict:
    """Train the head on (task, target) pairs.

    Each batch dict from the loader must have keys:
        input_ids, attention_mask         — for the head's forward
        targets: list[ArchTargets]        — typed teacher targets
    """
    if spec is None:
        spec = TRAIN
    out_dir = _ensure_dir(out_dir)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=spec.sft_lr,
        weight_decay=0.01,
    )

    # Linear warmup → cosine decay. Per-epoch optim.step count is
    # floor(B/A) in-loop + 1 trailing flush iff B%A != 0, i.e. ceil(B/A).
    # N33 fix: old formula `B*E//A` under-counted by up to E (one missed
    # flush per epoch), so the final 3–5 steps ran past num_training_steps
    # → cosine bottomed out at LR≈0 → wasted compute.
    A = max(1, spec.sft_grad_accum)
    L = len(train_loader)
    per_epoch_optim = (L + A - 1) // A
    total_steps = max(1, per_epoch_optim * spec.sft_epochs)
    if spec.sft_max_steps is not None:
        # max_steps caps the global per-BATCH counter. If it spans
        # `e_full` full epochs + `rem` batches of the next, actual
        # scheduler.step count is:
        #   e_full × per_epoch_optim          (each full epoch + flush)
        # + ceil(rem / A)                     (partial epoch + max_steps flush)
        # The naive ceil(M/A) under-counts by `e_full` (causing the
        # final LR=0 wasted updates).
        M = spec.sft_max_steps
        e_full = M // L
        rem = M % L
        capped = e_full * per_epoch_optim + (rem + A - 1) // A
        total_steps = min(total_steps, max(1, capped))
    warmup_steps = max(1, int(total_steps * spec.sft_warmup_ratio))
    try:
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
        )
        print(f"[sft] cosine schedule: warmup={warmup_steps} / total={total_steps}")
    except Exception as e:
        print(f"[sft] schedule unavailable ({e}); using constant LR.")
        scheduler = None

    history: list[dict] = []
    best_eval_loss: float | None = None
    step = 0
    t0 = time.time()

    def _flush_grads(reason: str) -> None:
        """Apply pending accumulated gradients to params and zero them.

        Called at epoch boundaries / max_steps cutoff / training end so
        the trailing `n_batches % sft_grad_accum` batches in each epoch
        don't have their gradient silently dropped (it sat in `.grad`
        but never reached `optim.step()`).
        """
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not any(p.grad is not None for p in trainable):
            return  # nothing to flush
        # N17 guard: don't NaN-poison the params on a corrupt residual.
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in trainable):
            print(f"[sft] _flush_grads({reason}): non-finite grad detected; "
                  f"discarding residual to preserve params.", flush=True)
            optim.zero_grad(set_to_none=True)
            return
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optim.step()
        if scheduler is not None:
            scheduler.step()
        optim.zero_grad(set_to_none=True)

    for epoch in range(spec.sft_epochs):
        if hasattr(train_loader.dataset, "reshuffle"):
            train_loader.dataset.reshuffle()  # type: ignore[attr-defined]

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            targets = batch["targets"]

            head_out = model(input_ids=input_ids, attention_mask=attn)
            comp = sft_loss_batch(head_out, targets, spec)
            loss = comp["total"] / spec.sft_grad_accum

            # N17 guard 1: non-finite loss → don't backward (which would
            # propagate NaN/Inf grads into params).
            if not torch.isfinite(loss).all():
                print(f"[sft] step={step} loss not finite ({loss.item()}); "
                      f"skipping backward.", flush=True)
                optim.zero_grad(set_to_none=True)
                step += 1
                continue
            loss.backward()
            if (step + 1) % spec.sft_grad_accum == 0:
                # N17 guard 2: post-backward non-finite grad → skip step.
                # clip_grad_norm_ doesn't catch NaN (its comparison
                # silently fails to False), so we check explicitly.
                trainable = [p for p in model.parameters() if p.requires_grad]
                if any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for p in trainable):
                    print(f"[sft] step={step} non-finite grad after backward; "
                          f"skipping optim.step to preserve params.", flush=True)
                    optim.zero_grad(set_to_none=True)
                else:
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                    optim.step()
                    if scheduler is not None:
                        scheduler.step()
                    optim.zero_grad(set_to_none=True)

            if step % log_every == 0:
                rec = {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(comp["total"].detach().item()),
                    "loss_gate": float(comp["gate"].detach().item()),
                    "loss_role": float(comp["role"].detach().item()),
                    "loss_edge": float(comp["edge"].detach().item()),
                    "loss_seq":  float(comp["seq"].detach().item()),
                    "loss_model": (float(comp["model"].detach().item())
                                   if "model" in comp else 0.0),
                    "elapsed":   time.time() - t0,
                }
                history.append(rec)
                print(
                    f"[sft] step={step:>5} epoch={epoch} "
                    f"L={rec['loss']:.3f} "
                    f"g={rec['loss_gate']:.3f} r={rec['loss_role']:.3f} "
                    f"e={rec['loss_edge']:.3f} s={rec['loss_seq']:.3f} "
                    f"m={rec['loss_model']:.3f}"
                )
                if wandb_run is not None:
                    wandb_run.log(rec, step=step)

            if (step + 1) % spec.sft_save_every_n_steps == 0:
                save_head_checkpoint(model, out_dir, tag=f"step{step + 1}")
                print(f"[sft] saved intermediate checkpoint at step {step + 1}")

            step += 1
            if spec.sft_max_steps is not None and step >= spec.sft_max_steps:
                _flush_grads("max_steps")
                save_head_checkpoint(model, out_dir, tag="final")
                _write_history(out_dir, history)
                return {"history": history, "final_step": step}

        # End of epoch: flush any residual gradient from the trailing
        # `n_batches % sft_grad_accum` batches so their update is not
        # silently dropped. Without this, with grad_accum=2 we lose 1
        # batch's gradient per epoch boundary.
        _flush_grads(f"epoch{epoch}_end")

        if eval_loader is not None:
            eval_loss = evaluate_sft(model, eval_loader, spec, device=device)
            print(f"[sft] epoch {epoch} eval_loss={eval_loss:.4f}")
            history.append({"step": step, "epoch": epoch, "eval_loss": eval_loss})
            if wandb_run is not None:
                wandb_run.log({"eval_loss": eval_loss, "epoch": epoch}, step=step)
            if best_eval_loss is None or eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_head_checkpoint(model, out_dir, tag="best_eval")

        ep_ckpt = save_head_checkpoint(model, out_dir, tag=f"epoch{epoch}")
        print(f"[sft] saved epoch-{epoch} checkpoint → {ep_ckpt}")

    save_head_checkpoint(model, out_dir, tag="final")
    _write_history(out_dir, history)
    return {"history": history, "final_step": step}


def evaluate_sft(
    model: ArchitectureHead,
    loader: DataLoader,
    spec: TrainSpec,
    device: str = "cuda",
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            targets = batch["targets"]
            head_out = model(input_ids=input_ids, attention_mask=attn)
            comp = sft_loss_batch(head_out, targets, spec)
            losses.append(float(comp["total"].item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def _write_history(out_dir: Path, history: list[dict]) -> None:
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


__all__ = [
    "evaluate_sft",
    "load_head_checkpoint",
    "save_head_checkpoint",
    "sft_loss_batch",
    "sft_loss_single",
    "train_sft",
]
