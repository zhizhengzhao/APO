"""Architecture-level GRPO trainer (v3 typed distributions, no KL).

Per (task, head) pair:

  1. head(task) → typed logits (gate / role / edge / seq)
  2. For g = 1..G, sample one ConcreteArch from those logits
  3. For each sampled arch, run the executor (DeepSeek workers) to get a reward
  4. advantage_g = (reward_g - mean_g(reward)) / std_g(reward)         (group-relative baseline)
  5. log_pi_g = log_prob_joint(logits, arch_g)
                = log_Bern(gates) + log_Cat(roles) + log_Bern(edges) + log_PL(seq)
  6. loss = -mean_g(advantage_g · log_pi_g) - α · entropy(logits)

Backprop only through the head; executor / sampler are detached (the sampled
architectures are constants from autograd's POV; the gradients flow through
the *log probabilities* evaluated at those constants).

There is NO KL term — the head is a fresh policy planner, not a language
model. Entropy bonus prevents premature mode collapse.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from ..architecture.sampler import (
    ConcreteArch,
    log_prob_joint,
    sample_arch,
)
from ..architecture.spec import ArchLogits
from ..config import ARCH, TRAIN, ArchSpec, TrainSpec
from ..executor.multi_agent import MultiAgentExecutor
from ..head.model import ArchitectureHead
from ..reward.compute import compute_reward


# ---------------------------------------------------------------------------
# Data container for a GRPO mini-batch
# ---------------------------------------------------------------------------

@dataclass
class GRPOBatch:
    task_texts: list[str]
    gold_answers: list[str]
    task_samples: list[object] | None = None  # optional list of TaskSample for graders


# ---------------------------------------------------------------------------
# Entropy of the typed distribution (used as exploration bonus)
# ---------------------------------------------------------------------------

def entropy_typed(head_out: dict[str, torch.Tensor], spec: ArchSpec | None = None) -> torch.Tensor:
    """Sum-of-typed-entropies, averaged over the batch.

    For PL we use the marginal entropy of the *first draw* — exact PL entropy
    has a closed form but is O(N!) without dynamic programming. The marginal
    is a sound regularizer that prevents seq_scores from collapsing.
    """
    if spec is None:
        spec = ARCH
    N = spec.n_max

    # gate: H(Bern(p))
    g_logits = head_out["gate_logits"]                    # [B, N]
    g_p = torch.sigmoid(g_logits)
    h_g = -(
        g_p * torch.log(g_p.clamp_min(1e-9))
        + (1 - g_p) * torch.log((1 - g_p).clamp_min(1e-9))
    ).sum(dim=-1)                                         # [B]

    # role: H(Cat(softmax(role_logits)))
    log_q = F.log_softmax(head_out["role_logits"], dim=-1)
    q = log_q.exp()
    h_q = -(q * log_q).sum(dim=-1).sum(dim=-1)            # [B]

    # edge: H(Bern(p)) over non-diagonal positions
    e_logits = head_out["edge_logits"]                    # [B, N, N]
    e_p = torch.sigmoid(e_logits)
    h_e = -(
        e_p * torch.log(e_p.clamp_min(1e-9))
        + (1 - e_p) * torch.log((1 - e_p).clamp_min(1e-9))
    )
    eye = torch.eye(N, device=h_e.device, dtype=torch.bool).unsqueeze(0)
    h_e = h_e.masked_fill(eye, 0.0).sum(dim=(-1, -2))     # [B]

    # seq: marginal entropy of softmax(seq_scores) as a PL surrogate
    log_s = F.log_softmax(head_out["seq_scores"], dim=-1)
    s = log_s.exp()
    h_s = -(s * log_s).sum(dim=-1)                        # [B]

    return (h_g + h_q + h_e + h_s).mean()


# ---------------------------------------------------------------------------
# One GRPO step
# ---------------------------------------------------------------------------

def grpo_step(
    model: ArchitectureHead,
    tokenizer,
    batch: GRPOBatch,
    executor: MultiAgentExecutor,
    spec: TrainSpec | None = None,
    device: str = "cuda",
    arch_spec: ArchSpec | None = None,
) -> dict:
    """One GRPO update on a small batch of tasks. Returns logs."""
    if spec is None:
        spec = TRAIN
    if arch_spec is None:
        arch_spec = ARCH

    enc = tokenizer(
        batch.task_texts,
        padding=True,
        truncation=True,
        max_length=spec.tokenizer_max_len,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    # ---- forward (differentiable) ------------------------------------------
    head_out = model(input_ids=input_ids, attention_mask=attn)
    B = head_out["gate_logits"].shape[0]
    G = spec.grpo_group_size

    # ---- sample G architectures per task (no_grad on sampling) -------------
    sampled: list[list[ConcreteArch]] = [[None] * G for _ in range(B)]  # type: ignore[list-item]
    rewards = torch.zeros(G, B, device=device)

    with torch.no_grad():
        for b in range(B):
            logits_b_const = ArchLogits(
                gate_logits=head_out["gate_logits"][b].detach().to("cpu").float(),
                role_logits=head_out["role_logits"][b].detach().to("cpu").float(),
                edge_logits=head_out["edge_logits"][b].detach().to("cpu").float(),
                seq_scores=head_out["seq_scores"][b].detach().to("cpu").float(),
            )
            for g in range(G):
                arch = sample_arch(logits_b_const, arch_spec, deterministic=False)
                sampled[b][g] = arch
                trace = executor.run(batch.task_texts[b], arch)
                gold = batch.gold_answers[b]
                ts = batch.task_samples[b] if batch.task_samples is not None else None
                r = compute_reward(trace, gold, spec, task_sample=ts)
                rewards[g, b] = r.total

    # ---- advantage (group-relative) ----------------------------------------
    # Use unbiased=False so that G==1 returns std=0 (not NaN from N-1 division)
    # plus clamp_min to avoid divide-by-zero when all rewards in a group tie.
    # G==1 is degenerate (no group baseline) and gives zero advantage; we still
    # let the entropy bonus train the head, but training is much weaker — caller
    # should use G >= 2 (default 4).
    baseline = rewards.mean(dim=0, keepdim=True)                              # [1, B]
    std = rewards.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)    # [1, B]
    advantage = ((rewards - baseline) / std).detach()                          # [G, B]

    # ---- log_pi (differentiable through head logits) -----------------------
    log_pis = torch.zeros(G, B, device=device)
    for b in range(B):
        for g in range(G):
            arch = sampled[b][g]
            # Move arch tensors to device for log_prob compute
            arch_dev = ConcreteArch(
                active_mask=arch.active_mask.to(device),
                roles=arch.roles.to(device),
                edges=arch.edges.to(device),
                sequence=arch.sequence.to(device),
            )
            logits_b = ArchLogits(
                gate_logits=head_out["gate_logits"][b],
                role_logits=head_out["role_logits"][b],
                edge_logits=head_out["edge_logits"][b],
                seq_scores=head_out["seq_scores"][b],
            )
            log_pis[g, b] = log_prob_joint(logits_b, arch_dev)

    # ---- losses ------------------------------------------------------------
    loss_pg = -(advantage * log_pis).mean()
    h = entropy_typed(head_out, arch_spec)
    loss = loss_pg - spec.grpo_entropy_weight * h

    return {
        "loss": loss,
        "loss_pg": float(loss_pg.detach().item()),
        "entropy": float(h.detach().item()),
        "reward_mean": float(rewards.mean().item()),
        "reward_std": float(rewards.std(unbiased=False).item()),
        "reward_max": float(rewards.max().item()),
        "reward_min": float(rewards.min().item()),
    }


# ---------------------------------------------------------------------------
# Top-level training loop
# ---------------------------------------------------------------------------

def train_grpo(
    model: ArchitectureHead,
    tokenizer,
    batches: Sequence[GRPOBatch],
    executor: MultiAgentExecutor,
    spec: TrainSpec | None = None,
    out_dir: str = "checkpoints/grpo",
    device: str = "cuda",
    log_every: int = 1,
    save_every: int = 25,
    wandb_run=None,
) -> dict:
    """Train the head with architecture-level GRPO over a sequence of batches.

    `batches` is an iterable of `GRPOBatch`; one element == one optimization
    step. Caller decides batching strategy.
    """
    if spec is None:
        spec = TRAIN
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.to(device)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=spec.grpo_lr,
    )

    history: list[dict] = []
    t0 = time.time()
    for step, batch in enumerate(batches):
        out = grpo_step(model, tokenizer, batch, executor, spec, device=device)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        optim.step()
        optim.zero_grad(set_to_none=True)

        rec = {k: float(v) if not isinstance(v, torch.Tensor) else float(v.detach().item())
               for k, v in out.items()}
        rec["step"] = step
        rec["elapsed"] = time.time() - t0
        history.append(rec)

        if step % log_every == 0:
            print(
                f"[grpo] step={step:>4} "
                f"L={rec['loss']:.3f} pg={rec['loss_pg']:.3f} "
                f"H={rec['entropy']:.2f} "
                f"r̄={rec['reward_mean']:.3f}±{rec['reward_std']:.3f}"
            )
            if wandb_run is not None:
                wandb_run.log(rec, step=step)

        if (step + 1) % save_every == 0:
            from .sft import save_head_checkpoint
            save_head_checkpoint(model, out_path, tag=f"grpo_step{step+1}")

    from .sft import save_head_checkpoint
    save_head_checkpoint(model, out_path, tag="grpo_final")

    return {"history": history, "final_step": len(history)}


__all__ = ["GRPOBatch", "entropy_typed", "grpo_step", "train_grpo"]
