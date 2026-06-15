"""Resume / restart contract tests for GRPO training.

When a training job crashes (preemption, OOM, network blip), `resume.pt`
in `out_dir/` must let the next invocation pick up where the previous
left off — same head weights, same optimizer state, correct next step.
At $8/step on the qwen3.7-max worker, getting resume wrong is the most
expensive engineering bug we can have.

These tests use a tiny dummy backbone (no model download) + MockWorker
(no API calls) so they're fast + free.
"""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from arch_policy import (
    ARCH,
    GRPOBatch,
    MockWorker,
    MultiAgentExecutor,
    train_grpo,
)
from arch_policy.head.model import HeadConfig


# ---------------------------------------------------------------------------
# Tiny dummy head (no Qwen download) — mirrors the public head API
# ---------------------------------------------------------------------------

class _DummyBackboneOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _DummyBackbone(nn.Module):
    def __init__(self, hidden: int = 32, vocab: int = 512) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.config = type("Cfg", (), {"hidden_size": hidden})()

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        return _DummyBackboneOutput(self.embed(input_ids))

    def eval(self): return self


class _DummyHead(nn.Module):
    """Light copy of ArchitectureHead that emits 4 typed logits over the
    same shapes the real head does. Avoids any HF model download."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.head_cfg = HeadConfig(head_hidden=64, n_head_layers=1, dropout=0.0)
        self.arch_spec = ARCH
        self.backbone_name = "dummy"
        self.backbone = _DummyBackbone(hidden=hidden)
        self._freeze_backbone = False

        N, R, d = ARCH.n_max, ARCH.k_roles, ARCH.d_latent
        self.body = nn.Sequential(
            nn.Linear(hidden, self.head_cfg.head_hidden), nn.GELU(),
        )
        self.agent_proj = nn.Linear(self.head_cfg.head_hidden, N * d)
        self.slot_emb = nn.Embedding(N, d)
        nn.init.normal_(self.slot_emb.weight, std=0.05)
        self.head_g = nn.Linear(d, 1)
        self.head_Q = nn.Linear(d, R)
        self.head_S = nn.Linear(d, 1)
        self.M = nn.Parameter(torch.zeros(d, d))
        nn.init.normal_(self.M, std=0.1 / math.sqrt(d))
        self.B = nn.Parameter(torch.zeros(R, R))
        nn.init.normal_(self.B, std=0.5)
        self.b0 = nn.Parameter(torch.zeros(1))

    @property
    def device(self): return next(self.parameters()).device

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        seq_lens = attention_mask.sum(dim=1) - 1
        idx = seq_lens.view(-1, 1, 1).expand(-1, 1, h.size(-1))
        pooled = h.gather(1, idx).squeeze(1)
        body = self.body(pooled)
        N, R, d = ARCH.n_max, ARCH.k_roles, ARCH.d_latent
        B = body.shape[0]
        U_raw = self.agent_proj(body).view(B, N, d)
        slot_idx = torch.arange(N, device=U_raw.device)
        U = U_raw + self.slot_emb(slot_idx).unsqueeze(0)

        gate_logits = self.head_g(U).squeeze(-1)
        role_logits = self.head_Q(U)
        seq_scores  = self.head_S(U).squeeze(-1)

        UM = torch.matmul(U, self.M)
        latent = torch.matmul(UM, U.transpose(-2, -1)) / math.sqrt(d)
        Q = torch.softmax(role_logits, dim=-1)
        sbm = torch.matmul(torch.matmul(Q, self.B), Q.transpose(-2, -1))
        edge_logits = latent + sbm + self.b0.view(1, 1, 1)
        eye = torch.eye(N, device=edge_logits.device, dtype=torch.bool)
        edge_logits = edge_logits.masked_fill(eye.unsqueeze(0), -1e9)

        return {
            "gate_logits": gate_logits,
            "role_logits": role_logits,
            "edge_logits": edge_logits,
            "seq_scores":  seq_scores,
        }

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _DummyTokenizer:
    """Bare minimum to satisfy train_grpo. Returns torch tensors."""
    def __call__(self, texts, padding=True, truncation=True,
                 max_length=256, return_tensors="pt"):
        ids = torch.zeros(len(texts), 4, dtype=torch.long)
        for i, t in enumerate(texts):
            for j, ch in enumerate(t[:4]):
                ids[i, j] = ord(ch) % 512
        mask = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": mask}


def _make_batches(n_steps: int = 4, B: int = 2) -> list[GRPOBatch]:
    """A short tasks list, replicated into N batches of size B."""
    base = ["What is 2+2?", "Who is the president?", "Capital of France?", "Solve x+1=3"]
    out = []
    for s in range(n_steps):
        out.append(GRPOBatch(
            task_texts=base[:B],
            gold_answers=["4", "42", "42", "42"][:B],
        ))
    return out


def _make_model_optim():
    torch.manual_seed(0)
    return _DummyHead()


# ---------------------------------------------------------------------------
# Test 1: resume.pt round-trip preserves head + optimizer state byte-for-byte
# ---------------------------------------------------------------------------

def test_resume_roundtrip_preserves_model_and_optim_state(tmp_path):
    """After save_head_checkpoint + atomic resume.pt write, loading them
    back must yield IDENTICAL trainable parameter tensors AND optimizer
    moments — otherwise resumed training drifts away from a continuous
    run."""
    from arch_policy.training.sft import (
        load_head_checkpoint, save_head_checkpoint,
    )

    model = _make_model_optim()
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3,
    )

    # Take a few SGD-ish steps to populate Adam moments.
    for _ in range(3):
        x = model(torch.zeros(2, 4, dtype=torch.long),
                  torch.ones(2, 4, dtype=torch.long))
        loss = x["gate_logits"].sum() + x["role_logits"].sum()
        optim.zero_grad(); loss.backward(); optim.step()

    # Save head + resume.pt the same way train_grpo does.
    save_head_checkpoint(model, tmp_path, tag="step3")
    resume_pt = tmp_path / "resume.pt"
    tmp = resume_pt.with_suffix(".pt.tmp")
    torch.save({
        "ckpt_dir_name": "head_step3",
        "optim_state_dict": optim.state_dict(),
        "last_completed_step": 2,
    }, tmp)
    tmp.replace(resume_pt)

    saved_params = {k: v.detach().clone() for k, v in model.state_dict().items()}
    saved_optim = optim.state_dict()

    # Fresh model + optim, then load.
    model2 = _make_model_optim()
    optim2 = torch.optim.AdamW(
        [p for p in model2.parameters() if p.requires_grad], lr=1e-3,
    )
    load_head_checkpoint(model2, tmp_path / "head_step3")
    ck = torch.load(resume_pt, map_location="cpu", weights_only=False)
    optim2.load_state_dict(ck["optim_state_dict"])

    # Trainable params identical
    for k, v in model2.state_dict().items():
        if k not in saved_params:
            continue
        ref = saved_params[k]
        if not torch.equal(v, ref):
            diff = (v - ref).abs().max().item()
            assert diff < 1e-7, f"param {k!r} drifted after resume: max|Δ|={diff}"

    # Optimizer Adam moments (exp_avg / exp_avg_sq) identical
    s1 = saved_optim["state"]
    s2 = optim2.state_dict()["state"]
    assert set(s1) == set(s2), "optimizer param keys diverged after resume"
    for pk in s1:
        for mk in s1[pk]:
            v1 = s1[pk][mk]; v2 = s2[pk][mk]
            if isinstance(v1, torch.Tensor):
                assert torch.equal(v1, v2), (
                    f"opt state[{pk}][{mk}] drifted after resume"
                )

    # last_completed_step survives
    assert ck["last_completed_step"] == 2


# ---------------------------------------------------------------------------
# Test 2: end-to-end train_grpo crash + resume = picks up at correct step
# ---------------------------------------------------------------------------

def test_train_grpo_resume_starts_at_correct_step(tmp_path, monkeypatch):
    """Full train_grpo flow: run 2 steps with save_every=2, then re-invoke
    on same out_dir → must SKIP steps 0-1 and run only the remaining."""
    # Spy on grpo_step calls so we know which steps actually executed.
    import arch_policy.training.grpo as grpo_mod
    real_step = grpo_mod.grpo_step
    step_calls: list[int] = []
    def _counting_step(*args, **kwargs):
        # Record which batch this corresponds to by snooping the first
        # task text (we'll seed unique tasks per step below).
        batch = kwargs.get("batch", args[2] if len(args) > 2 else None)
        if batch is not None:
            step_calls.append(int(batch.task_texts[0]))
        return real_step(*args, **kwargs)
    monkeypatch.setattr(grpo_mod, "grpo_step", _counting_step)

    # Need unique tasks so we can attribute calls to step indices.
    n_steps = 4
    batches = [
        GRPOBatch(task_texts=[str(s), str(s) + "_"],
                  gold_answers=["x", "x"])
        for s in range(n_steps)
    ]

    model = _make_model_optim()
    tok = _DummyTokenizer()
    ex = MultiAgentExecutor(worker=MockWorker(), wall_clock_timeout_s=5.0,
                            max_llm_calls_per_trace=4)

    # ---- Session 1: complete steps 0, 1; checkpoint written ------------
    train_grpo(
        model=model, tokenizer=tok, batches=batches[:2],
        executor=ex, max_concurrent_runs=4, device="cpu",
        out_dir=str(tmp_path),
        save_every=2, log_every=1, inject_k=0,
    )
    s1_calls = list(step_calls)
    assert s1_calls == [0, 1], f"session 1 should run steps [0,1], got {s1_calls}"
    assert (tmp_path / "resume.pt").exists(), "resume.pt not written after save_every=2"

    # ---- Session 2: same out_dir → must resume past completed steps ----
    step_calls.clear()
    model2 = _make_model_optim()  # weights/optim will be REPLACED via load
    train_grpo(
        model=model2, tokenizer=tok, batches=batches,
        executor=ex, max_concurrent_runs=4, device="cpu",
        out_dir=str(tmp_path),
        save_every=2, log_every=1, inject_k=0,
    )
    s2_calls = list(step_calls)
    assert s2_calls == [2, 3], (
        f"session 2 should SKIP steps [0,1] and run [2,3], got {s2_calls}. "
        "Resume start_step calculation may be off."
    )


# ---------------------------------------------------------------------------
# Test 3: resume reconstructs head state matching the saved checkpoint
# ---------------------------------------------------------------------------

def test_resume_loads_head_weights_into_fresh_model(tmp_path):
    """The model object passed to the second train_grpo() invocation is
    fresh-init; the load path must OVERWRITE its weights with the saved
    ones, otherwise resumed training silently restarts from random."""
    n_steps = 2
    batches = [
        GRPOBatch(task_texts=[str(s), str(s) + "_"], gold_answers=["x", "x"])
        for s in range(n_steps)
    ]
    tok = _DummyTokenizer()
    ex = MultiAgentExecutor(worker=MockWorker(), wall_clock_timeout_s=5.0,
                            max_llm_calls_per_trace=4)

    # Session 1: train + save
    model = _make_model_optim()
    train_grpo(
        model=model, tokenizer=tok, batches=batches,
        executor=ex, max_concurrent_runs=4, device="cpu",
        out_dir=str(tmp_path),
        save_every=2, log_every=1, inject_k=0,
    )
    saved_head_g_weight = model.head_g.weight.detach().clone()

    # Session 2: fresh model with DIFFERENT init seed → if resume doesn't
    # actually load weights, head_g.weight would stay at the fresh-init
    # value, not the saved one.
    torch.manual_seed(42)  # different seed
    model2 = _DummyHead()
    fresh_head_g_weight = model2.head_g.weight.detach().clone()
    assert not torch.equal(saved_head_g_weight, fresh_head_g_weight), (
        "test setup broken: fresh init coincides with saved weights"
    )

    train_grpo(
        model=model2, tokenizer=tok, batches=batches,
        executor=ex, max_concurrent_runs=4, device="cpu",
        out_dir=str(tmp_path),
        save_every=2, log_every=1, inject_k=0,
    )
    # With all batches already completed (start_step >= len(batches)),
    # session 2 should NOT take any further training steps → the model
    # weights should equal the saved ones, NOT the fresh init.
    after_resume = model2.head_g.weight.detach()
    assert torch.allclose(after_resume, saved_head_g_weight, atol=1e-6), (
        f"resume loaded wrong head_g weights: "
        f"max|Δ vs saved|={(after_resume - saved_head_g_weight).abs().max().item()}, "
        f"max|Δ vs fresh|={(after_resume - fresh_head_g_weight).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Test 4: corrupt resume.pt → fall back to fresh start, not crash
# ---------------------------------------------------------------------------

def test_corrupt_resume_pt_falls_back_to_fresh_start(tmp_path):
    """If resume.pt is half-written or its referenced ckpt dir is missing,
    train_grpo must print a warning and start from step 0 — NEVER crash
    a fresh GRPO run because of stale junk."""
    # Write a deliberately bad resume.pt (points to nonexistent ckpt).
    torch.save({
        "ckpt_dir_name": "head_nonexistent",
        "optim_state_dict": {},
        "last_completed_step": 7,
    }, tmp_path / "resume.pt")

    model = _make_model_optim()
    tok = _DummyTokenizer()
    ex = MultiAgentExecutor(worker=MockWorker(), wall_clock_timeout_s=5.0,
                            max_llm_calls_per_trace=4)
    batches = [
        GRPOBatch(task_texts=["a", "b"], gold_answers=["x", "x"])
        for _ in range(2)
    ]
    # Must not raise.
    out = train_grpo(
        model=model, tokenizer=tok, batches=batches,
        executor=ex, max_concurrent_runs=4, device="cpu",
        out_dir=str(tmp_path),
        save_every=2, log_every=1, inject_k=0,
    )
    # History should have entries for steps 0 and 1 (fresh start), not skip them.
    rec_steps = sorted({rec["step"] for rec in out["history"] if "step" in rec})
    assert rec_steps == [0, 1], (
        f"corrupt resume.pt should trigger fresh start; got history steps={rec_steps}"
    )
