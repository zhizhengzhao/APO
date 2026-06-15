"""Architecture-policy head: backbone + latent agent embedding + 4 typed heads.

Pipeline:

    backbone(task) → last-token hidden → MLP body → h ∈ R^{d_h}
        ↓ agent_proj + slot_emb
    U ∈ R^{N × d}   (per-slot agent latents)

    gate_logits[i]    = w_g · U[i]                                   [N]
    role_logits[i,r]  = (W_Q · U[i])[r]                              [N, R]
    seq_scores[i]     = w_s · U[i]                                   [N]
    edge_logits[i,j]  = U[i]^T M U[j]/√d  +  Q[i]^T B Q[j]  +  b0    [N, N]
                       (latent-space + role-pair SBM; diag → -1e9)

where Q = softmax(role_logits). All 4 heads share U.

Trainability modes (set via `freeze_backbone` + `lora_rank`):

  - lora_rank > 0           → LoRA on backbone + full heads (recommended; 24GB+)
  - freeze_backbone=True    → heads only (~1M trainable params)
  - both False              → full backbone FT (needs gradient_checkpointing
                              on a single 80GB card for ≥4B backbones)

5th typed head — `head_M` (per-slot model choice) — is instantiated only
when `ArchSpec.n_models > 1`. With a single model it is absent and forward
emits no `model_logits`, so the head is byte-identical to the 4-head setup
and pre-model-dim checkpoints load cleanly. SFT does not supervise it;
GRPO explores model assignment from a uniform prior. (The earlier
`head_synth` per-task synth-model head is not reintroduced — Synth runs
on a fixed worker.)
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch
import torch.nn as nn

from ..config import ARCH, MODEL, ArchSpec


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
        backbone_name: HF model id (e.g. "Qwen/Qwen3-4B").
        arch_spec: tensor dimension layout (n_max, k_roles, d_latent).
        head_cfg: head MLP config (hidden size, layers, dropout, init scales).
        freeze_backbone: if True, backbone receives no gradients.
        lora_rank: if > 0, PEFT LoRA adapters on backbone; overrides
            freeze_backbone semantics (LoRA is itself trainable).
        lora_alpha / lora_dropout: standard PEFT LoRA hyperparams.
        gradient_checkpointing: trade compute for activation memory; needed
            for full FT of ≥4B backbones on single 80GB cards.
        torch_dtype: backbone dtype. Head MLPs default to fp32 (nn.Linear's
            default) and `forward` casts `last_hidden` to the body's
            current dtype, so head compute follows whatever dtype
            `self.body` parameters are in — if you `head.to(bfloat16)`
            you'll also be in bf16. Recommend leaving head in fp32.
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
        # transformers 5.x renamed `torch_dtype` → `dtype`. Probe and pick
        # whichever the installed version accepts.
        try:
            from inspect import signature
            kw = "dtype" if "dtype" in signature(AutoModel.from_pretrained).parameters else "torch_dtype"
        except Exception:
            kw = "torch_dtype"
        # Flash-attention 2 is 5-10x more memory-efficient than the default
        # eager attention on large B × seq² × heads tensors. Without this,
        # B=16 + seq=2048 OOMs immediately on a 80GB A100 even with
        # gradient_checkpointing on. Try flash first; fall back to sdpa,
        # then eager, if the installed transformers/torch combo doesn't
        # support flash for this backbone.
        load_kwargs = {kw: torch_dtype_obj, "trust_remote_code": True}
        backbone = None
        for impl in ("flash_attention_2", "sdpa", None):
            try:
                self.backbone = AutoModel.from_pretrained(
                    backbone_name,
                    **load_kwargs,
                    **({"attn_implementation": impl} if impl else {}),
                )
                backbone = self.backbone
                print(f"[head] backbone loaded with attn_implementation="
                      f"{impl!r}", flush=True)
                break
            except (ImportError, ValueError) as e:
                print(f"[head] attn_implementation={impl!r} unavailable "
                      f"({type(e).__name__}: {str(e)[:80]}); falling back",
                      flush=True)
                continue
        if backbone is None:
            # Last-ditch: bare load (transformers picks default)
            self.backbone = AutoModel.from_pretrained(
                backbone_name, **load_kwargs,
            )

        # Three trainability modes (LoRA / frozen / full FT). Heads are always
        # fully trainable regardless.
        self._lora_rank = lora_rank
        if lora_rank > 0:
            try:
                from peft import LoraConfig, get_peft_model, TaskType
            except ImportError as e:
                raise ImportError(
                    "peft not installed. Run `pip install peft`."
                ) from e
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
            self._freeze_backbone = False  # LoRA adapters are trainable
        elif freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            self._freeze_backbone = True
        else:
            self._freeze_backbone = False

        # Gradient checkpointing must be enabled after load + before first
        # forward. We gate on POST-LoRA `_freeze_backbone` so LoRA paths
        # still get checkpointing (the raw `freeze_backbone` arg would skip
        # it for LoRA, causing silent OOM on big backbones).
        if gradient_checkpointing and not self._freeze_backbone:
            try:
                if hasattr(self.backbone, "gradient_checkpointing_enable"):
                    self.backbone.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False},
                    )
                if hasattr(self.config, "use_cache"):
                    self.config.use_cache = False
                if hasattr(self.backbone, "enable_input_require_grads"):
                    self.backbone.enable_input_require_grads()
            except Exception as e:
                print(f"[head] gradient checkpointing setup warning: {e}")

        # Resolve hidden_size — multimodal configs (Qwen3.5+) nest it inside
        # text_config / language_model. Try several candidates.
        def _resolve_hidden(cfg) -> int | None:
            for path in (
                lambda c: getattr(c, "hidden_size", None),
                lambda c: getattr(getattr(c, "text_config", None), "hidden_size", None),
                lambda c: getattr(getattr(c, "language_model", None), "hidden_size", None),
            ):
                v = path(cfg)
                if v is not None:
                    return int(v)
            return None
        hidden = (
            _resolve_hidden(self.config)
            or _resolve_hidden(getattr(self.backbone, "config", self.config))
        )
        if hidden is None:
            raise RuntimeError(
                f"Could not resolve hidden_size from {backbone_name}'s config "
                "(checked top-level, text_config, language_model, and backbone.config)."
            )
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

        # Per-slot heads operate on each U[i].
        self.head_g = nn.Linear(d, 1)
        self.head_Q = nn.Linear(d, R)
        self.head_S = nn.Linear(d, 1)
        # Per-slot model choice (5th typed head). Instantiated ONLY when
        # the model pool has >1 entry; with a single model the dimension
        # is absent and forward emits no model_logits — byte-identical to
        # the 4-head setup, and pre-model-dim checkpoints load with no
        # missing keys.
        self.head_M = nn.Linear(d, arch_spec.n_models) if arch_spec.n_models > 1 else None

        # Edge bilinear: Latent Space Model (Hoff '02) for agent affinity.
        self.M = nn.Parameter(torch.zeros(d, d))
        nn.init.normal_(self.M, std=head_cfg.bilinear_init_scale / math.sqrt(d))

        # Edge SBM: Stochastic Block Model on roles (Holland '81). B is
        # (R, R) and contributes `Q B Q^T` to edge_logits, where
        # `Q = softmax(role_logits)` is the MARGINAL role distribution,
        # NOT the sampled role one-hot. So B's effect on edge probability
        # is averaged over the role posterior — strictly:
        #   E_{r_i ~ Q[i], r_j ~ Q[j]} [B[r_i, r_j]]   (i ≠ j)
        # not the conditional B[sampled_i, sampled_j]. Soft-Q keeps
        # sample-time and log_prob-time consistent (both use the same
        # `Q`), preserving the unbiased on-policy estimator at the cost
        # of a per-sample interpretation of B. Read B as a *prior*
        # bias on role-pair edge likelihood, NOT as per-sample affinity.
        self.B = nn.Parameter(torch.zeros(R, R))
        nn.init.normal_(self.B, std=head_cfg.sbm_init_scale)

        # Global edge bias, init negative so a random-init head produces
        # sparse edges (~3-4 per 5-active arch, matching canonical
        # density), not the ~15-edge graphs a 0-init would give.
        self.b0 = nn.Parameter(torch.full((1,), -2.0))

        # Small per-head init so all 4 typed distributions start near-uniform;
        # task-conditional preferences are learned from SFT + GRPO signal.
        nn.init.normal_(self.agent_proj.weight, std=0.02)
        nn.init.zeros_(self.agent_proj.bias)
        typed_heads = [self.head_g, self.head_Q, self.head_S]
        if self.head_M is not None:
            typed_heads.append(self.head_M)
        for layer in typed_heads:
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
        """Hidden state of the LAST non-pad token, padding-side-agnostic.

        Naive `mask.sum(1) - 1` works for right-padding but lands inside
        the padding region for left-padded inputs (Qwen3 default for
        generation). Using cumsum == total finds the last 1 either way.
        """
        # Guard: all-zero attention_mask row → cum==total is True
        # everywhere → argmax=0 → silently pull padding token. Refuse.
        total = attention_mask.sum(dim=1, keepdim=True)             # [B, 1]
        if (total == 0).any():
            empty_rows = torch.nonzero(total.squeeze(-1) == 0).flatten().tolist()
            raise ValueError(
                f"_pool_last_token got all-zero attention_mask rows "
                f"{empty_rows[:5]}{'...' if len(empty_rows) > 5 else ''}; "
                f"caller passed empty inputs. Refusing to silently return "
                f"padding-token hidden state."
            )
        cum   = attention_mask.cumsum(dim=1)                        # [B, T]
        last_idx = (cum == total).long().argmax(dim=1).clamp(min=0)
        idx = last_idx.view(-1, 1, 1).expand(-1, 1, hidden_states.size(-1))
        return hidden_states.gather(1, idx).squeeze(1)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Returns the typed-head outputs plus inspection tensors:

          gate_logits   [B, N]
          role_logits   [B, N, R]
          edge_logits   [B, N, N]
          seq_scores    [B, N]
          model_logits  [B, N, n_models]   (only if n_models > 1)
          pooled        [B, H]      (inspection)
          agent_emb     [B, N, d]   (inspection)
        """
        from contextlib import nullcontext
        ctx = torch.no_grad() if self._freeze_backbone else nullcontext()
        with ctx:
            backbone_out = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
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
        N, d = self.arch_spec.n_max, self.arch_spec.d_latent
        B = h.shape[0]

        U_raw = self.agent_proj(h).view(B, N, d)       # [B, N, d]
        slot_idx = torch.arange(N, device=U_raw.device)
        U = U_raw + self.slot_emb(slot_idx).unsqueeze(0)   # [B, N, d]

        gate_logits = self.head_g(U).squeeze(-1)       # [B, N]
        role_logits = self.head_Q(U)                   # [B, N, R]
        seq_scores  = self.head_S(U).squeeze(-1)       # [B, N]

        # Edge logits = latent affinity + role-pair SBM + bias.
        UM = torch.matmul(U, self.M)                   # [B, N, d]
        latent_term = torch.matmul(UM, U.transpose(-2, -1)) / math.sqrt(d)
        Q = torch.softmax(role_logits, dim=-1)         # [B, N, R]
        QB = torch.matmul(Q, self.B)
        sbm_term = torch.matmul(QB, Q.transpose(-2, -1))   # [B, N, N]
        edge_logits = latent_term + sbm_term + self.b0.view(1, 1, 1)
        # Mask diagonal so self-loops can never be sampled and contribute
        # zero to all losses (sigmoid(-1e9) ≈ 0).
        eye = torch.eye(N, device=edge_logits.device, dtype=torch.bool)
        edge_logits = edge_logits.masked_fill(eye.unsqueeze(0), -1e9)

        out = {
            "gate_logits": gate_logits,
            "role_logits": role_logits,
            "edge_logits": edge_logits,
            "seq_scores":  seq_scores,
            "pooled":      pooled,
            "agent_emb":   U,
        }
        if self.head_M is not None:
            out["model_logits"] = self.head_M(U)   # [B, N, n_models]
        return out

    # ------------------------------------------------------------------
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_tokenizer(name: str = MODEL.head_model):
    from transformers import AutoTokenizer  # lazy import

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # Force right padding for safety; our pooler also handles left-padding
    # via cumsum, but right-padding keeps debug output more readable.
    tok.padding_side = "right"
    return tok


def to_arch_logits(out_dict: dict, batch_idx: int = 0):
    """Slice one batch row of the head's output dict into `ArchLogits`.

    Deferred import avoids a circular dep between `head` and `architecture`.
    """
    from ..architecture.spec import ArchLogits

    ml = out_dict.get("model_logits")
    return ArchLogits(
        gate_logits=out_dict["gate_logits"][batch_idx].detach(),
        role_logits=out_dict["role_logits"][batch_idx].detach(),
        edge_logits=out_dict["edge_logits"][batch_idx].detach(),
        seq_scores =out_dict["seq_scores"][batch_idx].detach(),
        model_logits=ml[batch_idx].detach() if ml is not None else None,
    )


__all__ = ["ArchitectureHead", "HeadConfig", "load_tokenizer", "to_arch_logits"]
