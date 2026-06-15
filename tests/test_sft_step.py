"""Single-step SFT smoke test using a tiny dummy backbone (no model download).

Verifies typed losses + a dummy head can drive loss down on a small
synthetic dataset.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from arch_policy import (
    ARCH,
    ArchTargets,
    encode_named_arch,
    full_library,
    sft_loss_batch,
    sft_loss_single,
)
from arch_policy.architecture.spec import ArchLogits
from arch_policy.head.model import HeadConfig


class _DummyBackboneOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _DummyBackbone(nn.Module):
    """Mimics HF AutoModel return shape so we can exercise the head pipeline
    without downloading a real Qwen model."""

    def __init__(self, hidden: int = 64, vocab: int = 1024) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.config = type("Cfg", (), {"hidden_size": hidden})()

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        x = self.embed(input_ids)
        return _DummyBackboneOutput(x)

    def eval(self):
        return self


class _DummyHead(nn.Module):
    """Mirror of ArchitectureHead's v3 typed-head structure with a synthetic
    backbone (no transformers dependency)."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.head_cfg = HeadConfig(head_hidden=64, n_head_layers=1, dropout=0.0)
        self.arch_spec = ARCH
        self.backbone_name = "dummy"
        self.backbone = _DummyBackbone(hidden=hidden)
        self._freeze_backbone = False

        N, R, d = ARCH.n_max, ARCH.k_roles, ARCH.d_latent

        self.body = nn.Sequential(
            nn.Linear(hidden, self.head_cfg.head_hidden),
            nn.GELU(),
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
        latent_term = torch.matmul(UM, U.transpose(-2, -1)) / math.sqrt(d)
        Q = torch.softmax(role_logits, dim=-1)
        QB = torch.matmul(Q, self.B)
        sbm_term = torch.matmul(QB, Q.transpose(-2, -1))
        edge_logits = latent_term + sbm_term + self.b0.view(1, 1, 1)
        eye = torch.eye(N, device=edge_logits.device, dtype=torch.bool)
        edge_logits = edge_logits.masked_fill(eye.unsqueeze(0), -1e9)

        return {
            "gate_logits": gate_logits,
            "role_logits": role_logits,
            "edge_logits": edge_logits,
            "seq_scores":  seq_scores,
        }


def test_sft_loss_components_finite():
    torch.manual_seed(0)
    N, R = ARCH.n_max, ARCH.k_roles
    logits = ArchLogits(
        gate_logits=torch.randn(N),
        role_logits=torch.randn(N, R),
        edge_logits=torch.randn(N, N),
        seq_scores =torch.randn(N),
    )
    eye = torch.eye(N, dtype=torch.bool)
    logits.edge_logits = logits.edge_logits.masked_fill(eye, -1e9)
    target = encode_named_arch(full_library(seed=0)[0])
    comp = sft_loss_single(logits, target)
    for k, v in comp.items():
        assert torch.isfinite(v), f"{k} not finite: {v}"
    # 4-loss design: total + 4 components, no model/synth
    assert set(comp.keys()) == {"gate", "role", "edge", "seq", "total"}


def test_dummy_sft_step_decreases_loss():
    torch.manual_seed(0)
    model = _DummyHead(hidden=32)
    lib = full_library(seed=0)
    targets = [encode_named_arch(a) for a in lib[:8]]
    input_ids = torch.randint(0, 1024, (8, 16))
    attn = torch.ones_like(input_ids)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(40):
        head_out = model(input_ids, attn)
        comp = sft_loss_batch(head_out, targets)
        loss = comp["total"]
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] - 0.5, (
        f"loss did not decrease enough: {losses[0]:.3f} -> {losses[-1]:.3f}"
    )


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed.append(fn.__name__)
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    if failed:
        print(f"\n{len(failed)} test(s) failed: {failed}")
        sys.exit(1)
    print("\nall sft-step tests pass.")
