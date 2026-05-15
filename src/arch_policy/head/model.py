"""Architecture-policy head built on a Qwen LM (v3.5 — backbone trainable).

Design (latent-agent-embedding + 4 typed heads):

    backbone(task) → context ∈ R^{H}        (last-token hidden state)
                       ↓ MLP body
                     h ∈ R^{d_h}
                       ↓ agent_proj
                     U_raw ∈ R^{N × d}      (split N agent slots × d_latent)
                       ↓ + slot_emb         (per-slot bias to break symmetry)
                     U ∈ R^{N × d}

    gate_logits[i]   = w_g · U[i]                                  → [N]
    role_logits[i,r] = (W_Q · U[i])[r]                             → [N, R]
    seq_scores[i]    = w_s · U[i]                                  → [N]
    edge_logits[i,j] = (U[i]^T M U[j])/sqrt(d) + Q[i]^T B Q[j] + b0  (i≠j; diag→-1e9)
                                                                   → [N, N]

    where Q = softmax(role_logits) is used for the SBM term.

The 4 heads share the agent embedding U — this is the key inductive bias.

Trainability modes (controlled by `freeze_backbone` and `lora_rank` flags):

  - freeze_backbone=True, lora_rank=0   : frozen backbone + trainable heads only
                                          (lightest; ~1M trainable params; was V3 default)
  - freeze_backbone=False, lora_rank=0  : full fine-tune (backbone + heads)
                                          (DEFAULT in V3.5; needs gradient checkpointing
                                          on a single 80GB card for 9B; consider LoRA)
  - freeze_backbone=False, lora_rank>0  : LoRA fine-tune of backbone + full heads
                                          (recommended for 9B+ on memory-constrained
                                          GPUs; LoRA wraps q/k/v/o/mlp linear layers)

Use `gradient_checkpointing=True` to trade compute for activation memory when
the backbone is unfrozen (essential for 9B full FT on 80GB).
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch
import torch.nn as nn

from ..config import ARCH, MODEL, ArchSpec, ModelSpec


@dataclass
class HeadConfig:
    head_hidden: int = 512
    n_head_layers: int = 2
    activation: str = "gelu"
    dropout: float = 0.05
    sbm_init_scale: float = 0.5    # how strongly role-pair preferences start
    bilinear_init_scale: float = 0.1  # initial M scale (small → near-zero edge logits at init)


def _activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"unknown activation {name}")


class ArchitectureHead(nn.Module):
    """Wraps a causal LM and adds a 4-typed-head architecture policy.

    Args:
        backbone_name: HF model id (e.g. "Qwen/Qwen3.5-9B").
        arch_spec: dimension layout for the output tensors.
        head_cfg: head MLP config.
        freeze_backbone: if True the backbone receives no gradients (lightest).
                         If False (DEFAULT in V3.5), backbone is trainable —
                         either full FT or LoRA depending on `lora_rank`.
        lora_rank: if > 0, wrap the backbone with PEFT LoRA at this rank
                   (and ignore freeze_backbone — LoRA always adds trainable
                   adapters on top of a frozen base). Recommended for 9B+ on
                   memory-constrained GPUs.
        lora_alpha: LoRA scaling (effective LR multiplier ≈ alpha/rank).
        lora_dropout: LoRA dropout for regularization.
        gradient_checkpointing: trade compute for activation memory; essential
                                for full FT 9B on a single 80GB card.
        torch_dtype: dtype for the backbone (head MLPs always fp32 for stability).
    """

    def __init__(
        self,
        backbone_name: str = MODEL.head_model,
        arch_spec: ArchSpec | None = None,
        head_cfg: HeadConfig | None = None,
        freeze_backbone: bool = False,
        lora_rank: int = 0,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        gradient_checkpointing: bool = False,
        torch_dtype: torch.dtype | str | None = None,
    ) -> None:
        super().__init__()
        if arch_spec is None:
            arch_spec = ARCH
        if head_cfg is None:
            head_cfg = HeadConfig()
        self.arch_spec = arch_spec
        self.head_cfg = head_cfg
        self.backbone_name = backbone_name

        if torch_dtype is None or torch_dtype == "auto":
            torch_dtype_obj = None
        elif isinstance(torch_dtype, str):
            torch_dtype_obj = getattr(torch, torch_dtype)
        else:
            torch_dtype_obj = torch_dtype

        from transformers import AutoConfig, AutoModel  # lazy import

        self.config = AutoConfig.from_pretrained(backbone_name, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(
            backbone_name,
            torch_dtype=torch_dtype_obj,
            trust_remote_code=True,
        )

        # ---- Trainability mode ------------------------------------------------
        # Three modes: (a) frozen backbone, (b) full FT, (c) LoRA on backbone.
        # Heads (body / agent_proj / slot_emb / head_g / head_Q / head_S /
        # M / B / b0) are ALWAYS fully trainable.
        self._lora_rank = lora_rank
        if lora_rank > 0:
            try:
                from peft import LoraConfig, get_peft_model, TaskType
            except ImportError as e:
                raise ImportError(
                    "peft not installed. Run `pip install peft` (or use the "
                    "project's requirements.txt)."
                ) from e
            # Cover the standard Qwen / Llama-style attention + MLP linears.
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
            lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=target_modules,
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.backbone = get_peft_model(self.backbone, lora_cfg)
            # PEFT auto-freezes everything not LoRA, so backbone base is frozen
            # and the LoRA adapters get gradients.
            self._freeze_backbone = False  # LoRA layers are trainable
        elif freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            self._freeze_backbone = True
        else:
            # Full fine-tune: leave backbone trainable.
            self._freeze_backbone = False

        # Gradient checkpointing must be enabled AFTER loading and BEFORE first
        # forward; transformers' API also disables use_cache automatically.
        self._gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing and not freeze_backbone:
            try:
                # PEFT-wrapped models also expose this method.
                if hasattr(self.backbone, "gradient_checkpointing_enable"):
                    self.backbone.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False},
                    )
                # Make sure use_cache is off (incompatible with checkpointing).
                if hasattr(self.config, "use_cache"):
                    self.config.use_cache = False
                # Trainable params with checkpointing need .requires_grad_() on
                # the backbone's input embeddings to propagate; transformers
                # provides this helper:
                if hasattr(self.backbone, "enable_input_require_grads"):
                    self.backbone.enable_input_require_grads()
            except Exception as e:
                print(f"[head] gradient checkpointing setup warning: {e}")

        hidden = int(getattr(self.config, "hidden_size"))
        N = arch_spec.n_max
        R = arch_spec.k_roles
        d = arch_spec.d_latent

        # MLP body: backbone hidden → head_hidden
        layers: list[nn.Module] = []
        prev = hidden
        for _ in range(head_cfg.n_head_layers):
            layers.append(nn.Linear(prev, head_cfg.head_hidden))
            layers.append(_activation(head_cfg.activation))
            if head_cfg.dropout > 0:
                layers.append(nn.Dropout(head_cfg.dropout))
            prev = head_cfg.head_hidden
        self.body = nn.Sequential(*layers)

        # Agent projection: head_hidden → N * d_latent (then reshape)
        self.agent_proj = nn.Linear(prev, N * d)

        # Slot embedding: per-slot bias to break symmetry
        self.slot_emb = nn.Embedding(N, d)
        nn.init.normal_(self.slot_emb.weight, std=0.05)

        # Per-agent heads (operate on each U[i])
        self.head_g = nn.Linear(d, 1)
        self.head_Q = nn.Linear(d, R)
        self.head_S = nn.Linear(d, 1)

        # Edge bilinear params (Latent Space Model)
        self.M = nn.Parameter(torch.zeros(d, d))
        nn.init.normal_(self.M, std=head_cfg.bilinear_init_scale / math.sqrt(d))

        # SBM (Stochastic Block Model) on roles
        self.B = nn.Parameter(torch.zeros(R, R))
        nn.init.normal_(self.B, std=head_cfg.sbm_init_scale)

        # Global edge bias
        self.b0 = nn.Parameter(torch.zeros(1))

        # Init the per-agent heads small for gentle starts
        nn.init.normal_(self.agent_proj.weight, std=0.02)
        nn.init.zeros_(self.agent_proj.bias)
        for layer in (self.head_g, self.head_Q, self.head_S):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ------------------------------------------------------------------
    def _pool_last_token(
        self,
        hidden_states: torch.Tensor,        # [B, T, H]
        attention_mask: torch.Tensor,       # [B, T]
    ) -> torch.Tensor:
        """Take hidden state of the last non-pad token for each row."""
        seq_lens = attention_mask.sum(dim=1) - 1
        seq_lens = seq_lens.clamp(min=0)
        idx = seq_lens.view(-1, 1, 1).expand(-1, 1, hidden_states.size(-1))
        return hidden_states.gather(1, idx).squeeze(1)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Returns a dict of typed head outputs (with leading batch dim B):

          gate_logits  [B, N]
          role_logits  [B, N, R]
          edge_logits  [B, N, N]
          seq_scores   [B, N]
          pooled       [B, H]      (for inspection)
          agent_emb    [B, N, d]   (for inspection)
        """
        ctx = torch.no_grad() if self._freeze_backbone else _NoOp()
        with ctx:
            backbone_out = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
            # PEFT models / some HF models expose the inner output via .base_model
            # but the .last_hidden_state attribute is consistent.
            last_hidden = (
                backbone_out.last_hidden_state
                if hasattr(backbone_out, "last_hidden_state")
                else backbone_out[0]
            )

        head_dtype = next(self.body.parameters()).dtype
        last_hidden = last_hidden.to(head_dtype)
        attention_mask_f = attention_mask.to(last_hidden.device)
        pooled = self._pool_last_token(last_hidden, attention_mask_f)  # [B, H]

        h = self.body(pooled)                          # [B, head_hidden]
        N, R, d = self.arch_spec.n_max, self.arch_spec.k_roles, self.arch_spec.d_latent
        B = h.shape[0]

        U_raw = self.agent_proj(h).view(B, N, d)       # [B, N, d]
        slot_idx = torch.arange(N, device=U_raw.device)
        U = U_raw + self.slot_emb(slot_idx).unsqueeze(0)   # [B, N, d]

        gate_logits = self.head_g(U).squeeze(-1)       # [B, N]
        role_logits = self.head_Q(U)                   # [B, N, R]
        seq_scores = self.head_S(U).squeeze(-1)        # [B, N]

        # ---- edge logits: latent + SBM ---------------------------------
        # latent term: U M U^T / sqrt(d)   shape [B, N, N]
        UM = torch.matmul(U, self.M)                   # [B, N, d]
        latent_term = torch.matmul(UM, U.transpose(-2, -1)) / math.sqrt(d)

        # SBM term: Q B Q^T   where Q = softmax(role_logits)
        Q = torch.softmax(role_logits, dim=-1)         # [B, N, R]
        QB = torch.matmul(Q, self.B)                   # [B, N, R]
        sbm_term = torch.matmul(QB, Q.transpose(-2, -1))  # [B, N, N]

        edge_logits = latent_term + sbm_term + self.b0.view(1, 1, 1)
        # Hard-mask the diagonal so self-loops can never be sampled, and
        # the diagonal contributes zero to all losses (sigmoid(-1e9) ≈ 0).
        eye = torch.eye(N, device=edge_logits.device, dtype=torch.bool)
        edge_logits = edge_logits.masked_fill(eye.unsqueeze(0), -1e9)

        return {
            "gate_logits": gate_logits,
            "role_logits": role_logits,
            "edge_logits": edge_logits,
            "seq_scores": seq_scores,
            "pooled": pooled,
            "agent_emb": U,
        }

    # ------------------------------------------------------------------
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _NoOp:
    """Context manager that does nothing — used to swap with `torch.no_grad`."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def load_tokenizer(name: str = MODEL.head_model):
    from transformers import AutoTokenizer  # lazy import

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


# ---------------------------------------------------------------------------
# Helper: extract one row of head output as `ArchLogits`
# ---------------------------------------------------------------------------

def to_arch_logits(out_dict: dict, batch_idx: int = 0):
    """Slice one row of the head's output dict and wrap as ArchLogits.

    This is a *deferred-import* helper to avoid a circular import between
    `head` and `architecture`.
    """
    from ..architecture.spec import ArchLogits

    return ArchLogits(
        gate_logits=out_dict["gate_logits"][batch_idx].detach(),
        role_logits=out_dict["role_logits"][batch_idx].detach(),
        edge_logits=out_dict["edge_logits"][batch_idx].detach(),
        seq_scores=out_dict["seq_scores"][batch_idx].detach(),
    )


__all__ = ["ArchitectureHead", "HeadConfig", "load_tokenizer", "to_arch_logits"]
