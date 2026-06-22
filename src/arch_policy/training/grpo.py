"""Architecture-level GRPO trainer.

Per (task, head) pair:

  1. head(task) → typed logits (gate / role / edge / seq).
  2. For g = 1..G, sample one ConcreteArch from those logits.
  3. Run the executor to get (correctness ∈ {0, 1}, n_calls). Reward = correctness.
  4. Advantage = shaped_advantage(correct, n_calls):  (see fn docstring)
       Tier 1 — raw: wrong = -1, correct = 1 + min-max bonus on n_calls
                     within correct sub-group (∈ [+1, +1+scale])
       Tier 2 — /σ per-task (no mean subtraction)
                σ=0 → -0.1 (all wrong) or 0 (all same correct)
       Properties: lex correct > wrong (sign preserved), cost ranks within
       correct only, rare-correct groups amplified by small σ.
  5. log_pi_g = log_prob_joint(logits, arch_g) over all 4 typed components.
  6. loss = −mean_g(advantage_g · log_pi_g) − Σ_c α_c · H_c(head_out).

Backprop only through the head; sampled architectures are detached
constants from autograd's POV. No KL — the head is a policy planner, not
a language model that needs language-prior protection. Per-head entropy
weights (see DEFAULT_ENTROPY_WEIGHTS) normalize each component's max
entropy so edge (~21 nats) doesn't drown out seq (~1.8 nats).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from ..architecture.sampler import (
    ConcreteArch,
    log_prob_joint,
    sample_arch,
)
from ..architecture.spec import ArchLogits
from ..config import ARCH, TRAIN, ArchSpec, TrainSpec
from ..executor.multi_agent import MultiAgentExecutor
from ..head.model import ArchitectureHead
from ..reward import compute_reward
from .advantage import shaped_advantage
from .entropy import DEFAULT_ENTROPY_WEIGHTS, entropy_typed


# ---------------------------------------------------------------------------
# Data container for a GRPO mini-batch
# ---------------------------------------------------------------------------

@dataclass
class GRPOBatch:
    task_texts: list[str]
    gold_answers: list[str]
    task_samples: list[object] | None = None  # optional list of TaskSample for graders


# ---------------------------------------------------------------------------
# `shaped_advantage` lives in `training.advantage`;
# `entropy_typed` + `DEFAULT_ENTROPY_WEIGHTS` live in `training.entropy`.
# Imported above, re-exported via __all__ at the end of this file.
# ---------------------------------------------------------------------------


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
    max_concurrent_runs: int = 32,
    inject_archs: list | None = None,
    inject_pool: list | None = None,
    inject_k: int = 0,
    inject_family_stratified: bool = False,
    inject_rng=None,
    reward_fn=None,
    arch_cache=None,
    arch_cache_step: int = 0,
) -> dict:
    """One GRPO update on a small batch of tasks. Returns logs.

    Architecture executions run in parallel via ThreadPoolExecutor — most
    workers are I/O-bound API calls so concurrency dominates.

    `arch_cache` (optional `ArchCache`): if provided, `_run_one` checks
    the cache before doing actual worker/judge calls. Cache hits skip
    the API entirely and return the previously computed reward + trace
    state. `arch_cache_step` is the current GRPO step index recorded in
    new cache entries (for forensics in arch_cache.jsonl).

    Three architecture-injection modes (mutually exclusive):

      A) `inject_archs`: fixed list, same K canonical archs every step.
      B) `inject_pool` + `inject_k`: per-step uniform-over-arch sample.
         BIASED toward high-variant families (use C for principled uniformity).
      C) `inject_family_stratified=True` + `inject_k`: per-step sample
         `inject_k` DISTINCT canonical families uniformly, then 1 variant
         from each. Equalizes BC pressure across all 42 canonical families.
         Default used in the released scripts.

    In all modes gradient flows through log_pi(arch) at the CURRENT head
    logits, so the head learns from injected archs even when its own
    policy would never sample them. Slightly biased estimator; bias → 0
    as the head's policy approaches the injection distribution.

    Args:
        inject_archs: mode A list (mutually exclusive with the others).
        inject_pool: mode B candidate pool of NamedArch / ConcreteArch.
        inject_k: number to sample per step. Must be ≤ pool size (mode B),
                  ≤ #families (mode C), and ≤ G in any mode.
        inject_family_stratified: enable mode C (ignores inject_pool).
        inject_rng: optional `random.Random` for reproducibility.
    """
    if spec is None:
        spec = TRAIN
    if arch_spec is None:
        # Source of truth is the head the model was actually built with —
        # NOT the global default ARCH (n_models=1). Using ARCH here silently
        # validated multi-model logits [N, n_models>1] against (N, 1) and
        # crashed every step of a multi-model run.
        arch_spec = getattr(model, "arch_spec", None) or ARCH

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

    # ---- prepare injected architectures (mode A / B / C) ------------------
    n_inject_modes = sum(int(x) for x in [
        inject_archs is not None,
        inject_pool is not None,
        inject_family_stratified,
    ])
    if n_inject_modes > 1:
        raise ValueError(
            "Pass exactly one of inject_archs (mode A) / inject_pool (mode B) / "
            "inject_family_stratified (mode C)."
        )
    selected: list = []
    if inject_archs is not None:
        # Mode A: same fixed set every step.
        selected = list(inject_archs)
    elif inject_pool is not None:
        # Mode B: random sample inject_k per step (uniform-over-arch).
        if inject_k <= 0 or inject_k > len(inject_pool):
            raise ValueError(
                f"inject_k={inject_k} must be in [1, len(inject_pool)={len(inject_pool)}]"
            )
        import random as _random
        rng = inject_rng if inject_rng is not None else _random
        selected = rng.sample(list(inject_pool), inject_k)
    elif inject_family_stratified:
        # Mode C: family-stratified — pick K families uniformly then 1
        # variant per family. Equalizes BC pressure across families.
        import random as _random
        from ..architecture.library import sample_inject_family_stratified
        rng = inject_rng if inject_rng is not None else _random.Random()
        if inject_k <= 0:
            raise ValueError(f"inject_k={inject_k} must be > 0 in mode C")
        selected = sample_inject_family_stratified(rng, inject_k)

    # Materialize NamedArchs → ConcreteArch. After dropping the
    # head_M / head_synth heads, NamedArch carries everything we need
    # (gate/role/edge/seq); no per-slot model / synth-model randomization
    # is required.
    inject_concrete: list[ConcreteArch] = []
    for item in selected:
        if hasattr(item, "to_concrete"):
            inject_concrete.append(item.to_concrete())
        elif isinstance(item, ConcreteArch):
            inject_concrete.append(item)
        else:
            raise TypeError(
                f"injected item must be NamedArch or ConcreteArch, "
                f"got {type(item).__name__}"
            )
    K = len(inject_concrete)
    if K > G:
        raise ValueError(f"K={K} > G={G} (inject_k must satisfy inject_k <= G)")

    # ---- sample G architectures per task (no_grad on sampling) -------------
    sampled: list[list[ConcreteArch]] = [[None] * G for _ in range(B)]  # type: ignore[list-item]
    rewards = torch.zeros(G, B, device=device)

    # Build per-task ArchLogits once, then sample G archs per task.
    per_task_logits: list[ArchLogits] = []
    with torch.no_grad():
        for b in range(B):
            _ml = head_out.get("model_logits")
            per_task_logits.append(ArchLogits(
                gate_logits=head_out["gate_logits"][b].detach().to("cpu").float(),
                role_logits=head_out["role_logits"][b].detach().to("cpu").float(),
                edge_logits=head_out["edge_logits"][b].detach().to("cpu").float(),
                seq_scores =head_out["seq_scores"][b].detach().to("cpu").float(),
                model_logits=(_ml[b].detach().to("cpu").float()
                              if _ml is not None else None),
            ))
        for b in range(B):
            # Slots 0..K-1 = injected canonical archs (same set for every task)
            for k in range(K):
                sampled[b][k] = inject_concrete[k]
            # Slots K..G-1 = on-policy samples
            for g in range(K, G):
                sampled[b][g] = sample_arch(per_task_logits[b], arch_spec,
                                            deterministic=False)

    # Execute all B*G architectures in parallel (I/O-bound API calls).
    # Collect (correct, n_calls) for shaped_advantage plus full
    # ExecutionTrace for verbose telemetry.
    correct = torch.zeros(G, B, device=device)
    n_calls = torch.zeros(G, B, device=device)
    traces: list[list] = [[None] * G for _ in range(B)]  # type: ignore[list-item]
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _cache_state_to_trace(state: dict, arch, task_text: str):
        """Inflate a cached trace_state dict into an object that quacks
        like ExecutionTrace for every field grpo.py reads downstream.

        We construct a real ExecutionTrace and write the cached state
        onto it (rather than a SimpleNamespace) so that any future
        attribute access not in `trace_to_state` returns the
        ExecutionTrace default rather than AttributeError-ing.
        """
        from ..executor.multi_agent import ExecutionTrace
        tr = ExecutionTrace(task=task_text, arch=arch)
        for k, v in state.items():
            try:
                setattr(tr, k, v)
            except Exception:  # noqa: BLE001
                pass
        return tr

    def _sentinel_trace(arch, b_idx: int, err_type: str, err_msg: str, tb: str):
        """Build an engineering-invalid trace so the rest of the batch survives.

        Marks `n_api_errors=1` — the GRPO eng-valid mask zeroes the
        advantage on these traces so infra noise contributes no gradient.
        Also writes the FULL error class + message + traceback into
        `trace.run_errors` so post-run forensics in `details.jsonl` shows
        what actually failed (no silent swallowing).
        See `tests/test_resilience.py::test_grpo_run_one_sentinel_is_masked`.
        """
        from ..executor.multi_agent import ExecutionTrace
        tr = ExecutionTrace(task=batch.task_texts[b_idx], arch=arch)
        tr.n_api_errors = 1
        # Short marker in synth_log for grep-ability; full structured
        # detail in run_errors so 05_analyze_grpo can group by err_type.
        tr.synth_log.append(f"[SENTINEL] {err_type}: {err_msg[:200]}")
        tr.run_errors.append({
            "kind": "run_one_uncaught",
            "type": err_type,
            "message": err_msg,
            "traceback": tb,
        })
        return tr

    def _run_one(b: int, g: int):
        arch = sampled[b][g]
        ts = batch.task_samples[b] if batch.task_samples is not None else None
        # ---- cache hit short-circuit -------------------------------------
        if arch_cache is not None and ts is not None:
            hit = arch_cache.get(ts.task_id, arch)
            if hit is not None:
                tr_view = _cache_state_to_trace(hit.trace_state, arch,
                                                batch.task_texts[b])
                return (b, g, float(hit.total), float(hit.correct),
                        int(hit.n_calls), tr_view)
        # ---- fresh execution --------------------------------------------
        try:
            trace = executor.run(batch.task_texts[b], arch)
            gold = batch.gold_answers[b]
            grader = reward_fn if reward_fn is not None else compute_reward
            r = grader(trace, gold, spec, task_sample=ts)
            total = float(r.total)
            correct_v = float(r.correctness)
            ncalls_v = int(r.n_calls)
            # ---- cache write --------------------------------------------
            # ONLY cache architecturally-attributable traces. Skip if this
            # was an infra event (either real API failure OR worker output
            # hit max_tokens cap) — that result is a function of infra
            # state at sample time, not a stable property of the (task,
            # arch) pair, and replaying it would just bake noise into
            # future steps' rewards.
            n_infra = (int(getattr(trace, "n_api_errors", 0))
                       + int(getattr(trace, "n_worker_truncations", 0)))
            if (arch_cache is not None and ts is not None
                    and n_infra == 0):
                from .arch_cache import CachedEntry, trace_to_state
                arch_cache.put(ts.task_id, arch, CachedEntry(
                    total=total, correct=int(correct_v), n_calls=ncalls_v,
                    n_active=int(r.n_active), n_edges=int(r.n_edges),
                    trace_state=trace_to_state(trace),
                    stored_at_step=int(arch_cache_step),
                ))
            return b, g, total, correct_v, ncalls_v, trace
        except Exception as e:  # noqa: BLE001 — must NEVER take down the GRPO step
            import traceback as _tb
            tb = _tb.format_exc()
            err_type, err_msg = type(e).__name__, str(e)
            print(f"[grpo] _run_one(b={b},g={g}) raised: "
                  f"{err_type}: {err_msg}\n{tb}", flush=True)
            tr = _sentinel_trace(arch, b, err_type, err_msg, tb)
            return b, g, 0.0, 0.0, 0, tr

    n_workers = min(max_concurrent_runs, B * G) if B * G > 0 else 1
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_run_one, b, g) for b in range(B) for g in range(G)]
        for fut in as_completed(futures):
            # fut.result() should never raise now — _run_one catches everything
            # and converts to a sentinel — but wrap one more time as belt+suspenders.
            try:
                b, g, total, c, nc, tr = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[grpo] future.result() unexpectedly raised: "
                      f"{type(e).__name__}: {e}", flush=True)
                continue
            rewards[g, b] = total
            correct[g, b] = c
            n_calls[g, b] = nc
            traces[b][g] = tr

    # ---- Engineering-invalid mask (attribution) -----------------------------
    # Architecture is the ONLY variable we want to affect the reward signal.
    # We mask traces where INFRA noise corrupted the final answer. Infra
    # noise comes in two flavors, both arch-untouchable:
    #
    #   - tr.n_api_errors  : worker chat exhausted its 6-retry budget
    #                        (real network / 5xx / gateway issue)
    #   - tr.n_worker_truncations : worker output hit max_new_tokens cap
    #                        (LLM verbosity beat the 8192 ceiling)
    #
    # MASK CASES:
    #   1. tr is None — _run_one always returns (its `except Exception`
    #      converts any failure into a sentinel trace), so the ONLY
    #      way traces[b][g] stays at its initial None is the belt-
    #      and-suspenders branch at fut.result(): if the worker thread
    #      raised an Exception subclass that _run_one's own except did
    #      not catch (e.g. an OOM during _sentinel_trace construction
    #      itself), the main thread's `except Exception` catches it
    #      and `continue`s without assigning traces[b][g]. A worker
    #      raising BaseException (KeyboardInterrupt / SystemExit)
    #      propagates past `except Exception` and aborts the whole
    #      step via the outer try/except in train_grpo. None traces
    #      thus represent infra failures we couldn't even structure.
    #   2. (n_api_errors > 0 OR n_worker_truncations > 0) AND not
    #      final_via_synth — infra event happened AND synth never
    #      closed, so final_answer came from heuristic_extract on a
    #      partial transcript that lost content. We can't separate
    #      "arch chose a bad path" from "infra ate part of the
    #      transcript", so we mask.
    #
    # We KEEP traces where infra event happened AND tr.final_via_synth
    # is True. Rationale (R37):
    #   - Failed turns leave `text=""` in trace.messages, which
    #     format_full_transcript renders as `[empty / skipped]` — a
    #     placeholder, NOT corrupted content. Synth's verdict is based
    #     on the OTHER agents' real content.
    #   - If Synth closed the verdict, the final_answer IS attributable
    #     to the architecture's design: the arch's multi-agent
    #     redundancy absorbed the 1-2 transient hiccups and still
    #     produced an answer Synth trusts.
    #   - The reward signal IS final_answer correctness. If that signal
    #     is reliable, the sample is valid signal about the arch.
    #   - Live data (3 steps, 384 traces): masking on n_api>0 dropped
    #     17% of traces, 89% of which were synth-closed and otherwise
    #     valid → throwing away ~10K usable traces over a 540-step run.
    #
    # IMPORTANT: hit_wall_clock and hit_call_cap are NOT masked. They are
    # the architecture's own choice (it picked a tool-heavy path that ran
    # over). The correctness of heuristic_extract on the partial transcript
    # drives the advantage directly. Bad architectures that frequently cap
    # out will accumulate low reward through the normal correctness signal,
    # which is exactly what we want GRPO to learn from.
    eng_valid = torch.ones(G, B, device=device, dtype=torch.bool)
    for b in range(B):
        for g in range(G):
            tr = traces[b][g]
            if tr is None:
                eng_valid[g, b] = False
                continue
            n_infra = (int(getattr(tr, "n_api_errors", 0))
                       + int(getattr(tr, "n_worker_truncations", 0)))
            if (n_infra > 0
                    and not getattr(tr, "final_via_synth", False)):
                eng_valid[g, b] = False
    n_eng_invalid = int((~eng_valid).sum().item())
    eng_invalid_rate = n_eng_invalid / max(1, G * B)

    # ---- advantage (shaped: lexicographic correct > wrong + z-score by -calls) ----
    # Pass eng_valid as valid_mask so σ is computed on the VALID subset
    # only — invalid sentinels would otherwise enter as "wrong" and
    # squeeze σ, artificially inflating the rare-correct amplification.
    advantage = shaped_advantage(
        correct=correct, n_calls=n_calls,
        cost_bonus_scale=spec.advantage_cost_bonus_scale,
        valid_mask=eng_valid,
    ).detach()

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
                model=arch.model.to(device) if arch.model is not None else None,
            )
            _ml = head_out.get("model_logits")
            logits_b = ArchLogits(
                gate_logits=head_out["gate_logits"][b],
                role_logits=head_out["role_logits"][b],
                edge_logits=head_out["edge_logits"][b],
                seq_scores =head_out["seq_scores"][b],
                model_logits=_ml[b] if _ml is not None else None,
            )
            log_pis[g, b] = log_prob_joint(logits_b, arch_dev)

    # Divide by # of engineering-VALID samples, not the full G*B.
    # Otherwise eng-invalid samples (advantage=0 via mask) shrink the
    # effective batch size in the denominator and silently down-scale the
    # gradient. With eng_invalid_rate at 5% this would shrink gradients
    # by 5%; at 20% by 20%.
    n_valid = max(1, int(eng_valid.sum().item()))
    loss_pg = -(advantage * log_pis).sum() / n_valid
    # entropy_typed already applies DEFAULT_ENTROPY_WEIGHTS per component;
    # `grpo_entropy_weight` is a global up/down-scaler over those weights.
    h = entropy_typed(head_out, arch_spec)
    loss = loss_pg - spec.grpo_entropy_weight * h

    # Inject vs on-policy split — diagnostic for BC vs explore balance.
    # If loss_pg_inject / loss_pg_onpolicy stays > 3 deep into training,
    # the head is still BC-bound; consider decaying K or raising α_H.
    if K > 0 and K < G:
        n_valid_inj = max(1, int(eng_valid[:K].sum().item()))
        n_valid_on  = max(1, int(eng_valid[K:].sum().item()))
        loss_pg_inject = -(advantage[:K] * log_pis[:K]).sum() / n_valid_inj
        loss_pg_onpolicy = -(advantage[K:] * log_pis[K:]).sum() / n_valid_on
    else:
        loss_pg_inject = torch.zeros((), device=device)
        loss_pg_onpolicy = loss_pg.detach()

    # Per-step monitoring of architecture statistics — critical for catching
    # single-agent collapse early.
    n_active_per_sample = torch.tensor(
        [[int(sampled[b][g].active_mask.sum().item()) for b in range(B)] for g in range(G)],
        dtype=torch.float32,
    )

    # Per-(task, sample) records — cheap to compute, dumped to details.jsonl
    # for offline analysis (unanimity distribution, role/edge frequencies,
    # n_calls spread within correct, etc.).
    correct_np = correct.detach().cpu().numpy().astype(int)        # [G, B]
    n_calls_np = n_calls.detach().cpu().numpy().astype(int)        # [G, B]
    n_active_np = n_active_per_sample.numpy().astype(int)          # [G, B]
    advantage_np = advantage.detach().cpu().numpy()                # [G, B]
    per_task_details = []
    # Per-step aggregates over all B*G archs.
    n_archs_via_synth = 0
    n_archs_hit_cycle_cap = 0
    n_archs_with_step_cap = 0
    n_archs_with_tool_timeout = 0
    n_archs_with_api_error = 0         # 6-retry exhausted (real infra)
    n_archs_with_worker_truncation = 0 # output cap hit (model verbosity)
    n_archs_with_search_stub = 0     # archs that had ≥1 search-tool stub
    search_stub_total = 0            # total stub returns across all archs
    n_archs_with_run_errors = 0      # archs with structured run_errors entries
    run_error_kinds: dict = {}       # {err_type: n_archs}
    tool_usage_total: dict = {}    # {tool_name: n_calls across all archs}
    tool_errors_total: dict = {}   # {tool_name: n_errors across all archs}
    # Per-step aggregate of turn-termination outcomes — see
    # ExecutionTrace.termination_breakdown for the 6-way breakdown.
    term_total: dict = {
        "submit_implicit": 0, "skipped_explicit": 0,
        "skip_hit_cap": 0,
        "skip_worker_error": 0, "skip_wall_clock": 0, "skip_empty_text": 0,
    }
    for b in range(B):
        samples = []
        for g in range(G):
            arch = sampled[b][g]
            tr = traces[b][g]
            active_slots = arch.active_mask.cpu().numpy().tolist()
            roles_list = arch.roles.cpu().numpy().tolist()
            # role of each ACTIVE slot (in slot order)
            active_roles = [int(roles_list[i]) for i, a in enumerate(active_slots) if a]
            # model id of each ACTIVE slot (parallel to active_roles); None
            # in single-model runs. This is THE telemetry for the model
            # dimension's research question: which model the head assigns
            # to which role.
            active_models = None
            if arch.model is not None:
                models_list = arch.model.cpu().numpy().tolist()
                active_models = [int(models_list[i]) for i, a in enumerate(active_slots) if a]
            seq_list = arch.sequence.cpu().numpy().tolist()  # speaking order
            edges_count = int(arch.edges.sum().item())
            # Per-arch verbose telemetry.
            tool_calls = dict(tr.tool_call_counts) if tr is not None else {}
            tool_errors = dict(tr.tool_error_counts) if tr is not None else {}
            tool_err_kinds = (
                dict(tr.tool_error_kinds)
                if tr is not None and hasattr(tr, "tool_error_kinds") else {}
            )
            python_exec_log = (
                list(tr.python_exec_log)
                if tr is not None and hasattr(tr, "python_exec_log") else []
            )
            via_synth = bool(tr.final_via_synth) if tr is not None else False
            hit_cyc = bool(tr.hit_cycle_cap) if tr is not None else False
            hit_wc  = bool(getattr(tr, "hit_wall_clock", False)) if tr is not None else False
            hit_cc  = bool(getattr(tr, "hit_call_cap", False)) if tr is not None else False
            n_api   = int(tr.n_api_errors) if tr is not None else 0
            n_wtrunc = (int(getattr(tr, "n_worker_truncations", 0))
                        if tr is not None else 0)
            n_acaps = int(tr.n_arch_caps_hit) if tr is not None else 0
            n_step_cap = int(tr.n_agents_hit_step_cap) if tr is not None else 0
            n_trunc = int(tr.n_tool_truncations) if tr is not None else 0
            in_tok = int(tr.total_input_tokens) if tr is not None else 0
            out_tok = int(tr.total_output_tokens) if tr is not None else 0
            wall_s = float(getattr(tr, "wall_seconds", 0.0)) if tr is not None else 0.0
            n_cyc = int(tr.n_cycles_run) if tr is not None else 0
            n_synth = int(tr.n_synth_calls) if tr is not None else 0
            # Protocol-compliance per (role, model) for this arch.
            pc_dict = (
                {k: dict(v) for k, v in tr.protocol_compliance.items()}
                if tr is not None else {}
            )
            n_skipped = int(tr.n_skipped_turns) if tr is not None else 0
            n_proto_fail = int(tr.n_protocol_fail_turns) if tr is not None else 0
            term_breakdown = (
                dict(tr.termination_breakdown)
                if tr is not None and hasattr(tr, "termination_breakdown")
                else {}
            )
            # aggregate into per-step totals
            if via_synth:
                n_archs_via_synth += 1
            if hit_cyc:
                n_archs_hit_cycle_cap += 1
            if n_step_cap > 0:
                n_archs_with_step_cap += 1
            if sum(tool_errors.values()) > 0:
                n_archs_with_tool_timeout += 1
            if n_api > 0:
                n_archs_with_api_error += 1
            if n_wtrunc > 0:
                n_archs_with_worker_truncation += 1
            # Search-tool stub telemetry (silent-degradation watch).
            stub_dict = (
                dict(getattr(tr, "search_stub_counts", {}) or {})
                if tr is not None else {}
            )
            stub_sum = sum(stub_dict.values())
            if stub_sum > 0:
                n_archs_with_search_stub += 1
            search_stub_total += stub_sum
            # Structured run-error rollup (engineering issues we caught).
            run_errs = list(getattr(tr, "run_errors", []) or []) if tr is not None else []
            if run_errs:
                n_archs_with_run_errors += 1
                for re_ in run_errs:
                    k = re_.get("type", "Unknown") if isinstance(re_, dict) else "Unknown"
                    run_error_kinds[k] = run_error_kinds.get(k, 0) + 1
            for tn, c in tool_calls.items():
                tool_usage_total[tn] = tool_usage_total.get(tn, 0) + c
            for tn, c in tool_errors.items():
                tool_errors_total[tn] = tool_errors_total.get(tn, 0) + c
            for k, v in term_breakdown.items():
                if k in term_total:
                    term_total[k] += int(v)
            # Per-arch silent-degradation + structured-error snapshots.
            # Without these on the per-arch level, post-mortem can only
            # tell "this step had N stubs" but not "which arch in which
            # task hit them" — important for diagnosing whether stubs
            # correlate with particular Researcher-using architectures.
            stub_counts = (
                dict(getattr(tr, "search_stub_counts", {}) or {})
                if tr is not None else {}
            )
            stub_sum = sum(stub_counts.values())
            run_errs = list(getattr(tr, "run_errors", []) or []) if tr is not None else []
            run_err_types = [re_.get("type", "Unknown") if isinstance(re_, dict)
                             else "Unknown" for re_ in run_errs]
            samples.append({
                "g": g,
                "correct": int(correct_np[g, b]),
                "n_calls": int(n_calls_np[g, b]),
                "n_active": int(n_active_np[g, b]),
                "active_roles": active_roles,
                "active_models": active_models,
                "sequence": seq_list,
                "edges_count": edges_count,
                "advantage": float(advantage_np[g, b]),
                "n_cycles_run": n_cyc,
                "n_synth_calls": n_synth,
                "n_in_tokens": in_tok,
                "n_out_tokens": out_tok,
                "wall_seconds": round(wall_s, 2),
                "final_via_synth": via_synth,
                "hit_cycle_cap": hit_cyc,
                "hit_wall_clock": hit_wc,
                "hit_call_cap":   hit_cc,
                "n_api_errors":         n_api,
                "n_worker_truncations": n_wtrunc,
                "n_arch_caps_hit":      n_acaps,
                "n_agents_hit_step_cap": n_step_cap,
                "n_tool_truncations":   n_trunc,
                "tool_call_counts": tool_calls,
                "tool_error_counts": tool_errors,
                "search_stub_counts": stub_counts,
                "search_stub_total":  stub_sum,
                "run_errors_count":   len(run_errs),
                "run_error_types":    run_err_types,
                "tool_error_kinds": tool_err_kinds,
                "python_exec_log":  python_exec_log,
                # Protocol-compliance breakdown per (role, model) — see
                # ExecutionTrace.protocol_compliance.
                "protocol_compliance": pc_dict,
                "n_skipped_turns": n_skipped,
                "n_protocol_fail_turns": n_proto_fail,
                # 6-way termination outcome counts for this trace — see
                # ExecutionTrace.termination_breakdown.
                "termination_breakdown": term_breakdown,
            })
        # Task-level summary. We surface global task_id + family so the
        # offline analyzer can do per-source / per-task drill-down. When
        # the batch lacks task_samples (unit tests) both default to None /
        # "(unknown)".
        correct_count = int(correct_np[:, b].sum())
        ts = batch.task_samples[b] if batch.task_samples is not None else None
        gtid = getattr(ts, "task_id", None) if ts is not None else None
        gfam = getattr(ts, "family", None) if ts is not None else None
        per_task_details.append({
            "task_idx": b,                          # batch-local 0..B-1
            "task_id":  gtid,                       # TaskSample.task_id
            "family":   gfam or "(unknown)",        # TaskSample.family
            "correct_count": correct_count,         # 0..G
            "correct_frac": correct_count / G,
            "n_calls_min":  int(n_calls_np[:, b].min()),
            "n_calls_mean": float(n_calls_np[:, b].mean()),
            "n_calls_max":  int(n_calls_np[:, b].max()),
            "n_active_mean": float(n_active_np[:, b].mean()),
            "samples": samples,
        })

    # Distribution of correct_count across the B tasks in this batch — the
    # key signal for "are tasks mostly unanimous (0/G or G/G) or mixed?".
    cc_dist = [int((correct.sum(dim=0) == k).sum().item()) for k in range(G + 1)]

    # Per-trace wall-clock distribution (real measured run() seconds). The
    # straggler signal: the step blocks on the slowest trace, so p90/max drive
    # step wall time. Full per-trace values are in details.jsonl["wall_seconds"].
    _wall = sorted(s["wall_seconds"] for t in per_task_details for s in t["samples"])
    def _wpct(p: float) -> float:
        return _wall[min(len(_wall) - 1, int(p / 100 * len(_wall)))] if _wall else 0.0

    return {
        "loss": loss,
        "loss_pg": float(loss_pg.detach().item()),
        # Inject vs on-policy split (BC vs explore monitoring; see comment
        # above the split computation). Both are 0 when K=0 or K=G.
        "loss_pg_inject":   float(loss_pg_inject.detach().item()),
        "loss_pg_onpolicy": float(loss_pg_onpolicy.detach().item()),
        "K_inject": K,
        "entropy": float(h.detach().item()),
        # rewards (binary correctness; cost is in advantage)
        "reward_mean": float(rewards.mean().item()),
        "reward_std": float(rewards.std(unbiased=False).item()),
        "reward_max": float(rewards.max().item()),
        "reward_min": float(rewards.min().item()),
        # correctness rate over the whole B*G batch
        "correct_rate": float(correct.mean().item()),
        # architecture statistics (monitor for collapse)
        "n_active_mean": float(n_active_per_sample.mean().item()),
        "n_active_min":  float(n_active_per_sample.min().item()),
        "n_calls_mean":  float(n_calls.mean().item()),
        "n_calls_min":   float(n_calls.min().item()),
        # per-trace wall-clock seconds (straggler signal; p90/max drive step time)
        "trace_wall_p50": _wpct(50),
        "trace_wall_p90": _wpct(90),
        "trace_wall_p99": _wpct(99),
        "trace_wall_max": (max(_wall) if _wall else 0.0),
        # advantage stats
        "adv_mean": float(advantage.mean().item()),
        "adv_max":  float(advantage.max().item()),
        "adv_min":  float(advantage.min().item()),
        # distribution of correct_count across B tasks: cc_dist[k] = how many
        # tasks had exactly k of G samples correct.
        # cc_dist[0]+cc_dist[G] = #unanimous tasks (signal-free for correctness).
        "correct_count_dist": cc_dist,
        "n_unanimous": cc_dist[0] + cc_dist[G],
        "n_mixed": B - (cc_dist[0] + cc_dist[G]),
        # ---- Budget-cap & tool-usage aggregates (per step, over B*G archs) ----
        # Counts of archs that bumped into an engineering budget; rising
        # values mean the caps are too tight (or the model is pathological).
        "n_archs_via_synth":         n_archs_via_synth,
        "n_archs_hit_cycle_cap":     n_archs_hit_cycle_cap,
        "n_archs_with_step_cap":     n_archs_with_step_cap,
        "n_archs_with_tool_timeout":      n_archs_with_tool_timeout,
        "n_archs_with_api_error":         n_archs_with_api_error,
        "n_archs_with_worker_truncation": n_archs_with_worker_truncation,
        # Silent-degradation watch. If `search_stub_total > 0` in a run
        # that uses Researcher-style archs, the search tools are NOT
        # returning real Serper results (either key missing or HTTP
        # errors). In strict_tools mode this can't happen — preflight
        # would have raised. With --no-strict_tools, this is the visible
        # signal you've been silently degrading and the reward signal
        # for Researcher families is corrupted.
        "n_archs_with_search_stub": n_archs_with_search_stub,
        "search_stub_total":        search_stub_total,
        # Structured run-error rollup. `n_archs_with_run_errors` is the
        # count of B*G archs whose _run_one() crashed (sentinel path).
        # `run_error_kinds` is the histogram of exception types so we
        # can tell "ConnectionResetError x40" apart from "ValueError x2".
        "n_archs_with_run_errors":  n_archs_with_run_errors,
        "run_error_kinds":          run_error_kinds,
        # Engineering-invalid samples (post-R37/R38 predicate):
        #   tr is None OR ((n_api_errors > 0 OR n_worker_truncations > 0)
        #                  AND NOT final_via_synth)
        # i.e. an infra event happened AND synth never closed a verdict.
        # Excluded from the advantage signal so architecture is the only
        # learning driver. Note this is STRICTLY weaker than the pre-R37
        # `n_api_errors > 0` predicate — synth-closed traces survive even
        # with infra hiccups because the placeholder-augmented transcript
        # still produced a trustworthy final_answer.
        # Target operational rate: < 3% of G*B samples (R37 live: 1-2%).
        "n_eng_invalid":   n_eng_invalid,
        "eng_invalid_rate": eng_invalid_rate,
        # Turn-termination breakdown across the WHOLE step (sums over all
        # G*B traces). submit_explicit + skipped_explicit are healthy
        # paths; the four skip_* are engineering / protocol issues to
        # drive toward <1% of total turns.
        "term_submit":          term_total["submit_implicit"],
        "term_skip_explicit":   term_total["skipped_explicit"],
        "term_skip_empty":      term_total["skip_empty_text"],
        "term_skip_worker_err": term_total["skip_worker_error"],
        "term_skip_wall_clock": term_total["skip_wall_clock"],
        "term_skip_hit_cap":    term_total["skip_hit_cap"],
        # Aggregate tool usage breakdown — which tools the head is actually
        # using, and how many of them errored / hit a timeout.
        "tool_usage_total":  tool_usage_total,
        "tool_errors_total": tool_errors_total,
        # full per-task details (callers can dump to JSONL for offline analysis)
        "per_task_details": per_task_details,
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
    inject_archs: list | None = None,
    inject_pool: list | None = None,
    inject_k: int = 0,
    inject_k_per_step: list[int] | None = None,
    inject_family_stratified: bool = False,
    inject_seed: int | None = None,
    max_concurrent_runs: int = 32,
    reward_fn=None,
    arch_cache=None,
) -> dict:
    """Train the head with architecture-level GRPO over a sequence of batches.

    `batches` is an iterable of `GRPOBatch`; one element == one optimization
    step. Caller decides batching strategy.

    Architecture injection (3 mutually exclusive modes):
      - `inject_archs`: fixed list, same K every step (mode A).
      - `inject_pool` + `inject_k`: pool, uniform sample K per step (mode B).
      - `inject_family_stratified=True` + `inject_k`: pick K distinct
        canonical families uniformly then 1 variant each (used in the released scripts).
    See `grpo_step` for the math.

    `inject_seed`: optional seed for the per-step inject sampler.
    `inject_k_per_step`: optional per-step K curriculum (overrides
    `inject_k`; length must equal number of steps).
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

    # ── Resume from prior interruption ─────────────────────────────────────
    # If `out_dir/resume.pt` exists, restore:
    #   - model trainable params (the latest grpo_stepN ckpt referenced inside)
    #   - optimizer state
    #   - last completed step index (to skip already-done batches)
    #   - history.json (so progress charts include the prior session)
    # The "resume.pt" pointer is rewritten every save_every steps; if the
    # process was killed mid-step, we lose at most save_every steps of work.
    resume_pt = out_path / "resume.pt"
    start_step = 0
    history: list[dict] = []
    resumed_inj_rng_state = None
    resumed_elapsed_baseline = 0.0
    import random as _random
    if resume_pt.exists():
        import json as _json
        # PHASE 1: load resume.pt. Narrow try/except: only catches the
        # things that justify "ignore resume, start fresh" (file corrupt
        # or unreadable). model/optim are untouched here.
        ck = None
        try:
            ck = torch.load(resume_pt, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001
            import traceback as _tb
            print(f"[grpo] resume.pt found but torch.load failed: "
                  f"{type(e).__name__}: {e}\n{_tb.format_exc()}\n"
                  f"  → starting from scratch (model/optim untouched).",
                  flush=True)
        if ck is not None:
            # PHASE 2: apply resume to model + optim.
            #
            # Two failure regimes — only the second is dangerous:
            #   (a) Pre-mutation failure (ckpt dir missing): model is
            #       still pristine, safe to fall back to fresh start.
            #   (b) Post-mutation failure (ckpt loaded partially, optim
            #       state corrupt): model is half-resumed, MUST raise.
            #       A silent start_step=0 reset here would let new ckpts
            #       overwrite the surviving ones.
            from .sft import load_head_checkpoint
            ckpt_tag_dir = out_path / ck.get("ckpt_dir_name", "")
            if not ckpt_tag_dir.exists():
                print(f"[grpo] resume.pt references missing ckpt dir "
                      f"{ckpt_tag_dir} → starting fresh (model untouched).",
                      flush=True)
            else:
                model_mutated = False
                try:
                    load_head_checkpoint(model, ckpt_tag_dir)
                    model_mutated = True   # any failure past here is dangerous
                    model.to(device)
                    optim.load_state_dict(ck["optim_state_dict"])
                    start_step = int(ck["last_completed_step"]) + 1
                    resumed_inj_rng_state = ck.get("inj_rng_state")
                    resumed_elapsed_baseline = float(ck.get("elapsed_baseline_s", 0.0))
                    # Restore arch_cache bernoulli RNG so reuse_prob<1.0
                    # resumes bit-identically.
                    cache_rng_state_restored = ck.get("cache_rng_state")
                    if (arch_cache is not None
                            and cache_rng_state_restored is not None):
                        try:
                            arch_cache._rng.setstate(
                                tuple(cache_rng_state_restored)
                                if not isinstance(cache_rng_state_restored, tuple)
                                else cache_rng_state_restored
                            )
                            print(f"[grpo] arch_cache rng restored (exact)",
                                  flush=True)
                        except Exception as e:  # noqa: BLE001
                            print(f"[grpo] arch_cache rng restore failed "
                                  f"({type(e).__name__}: {e}); RNG re-seeded "
                                  f"from --seed", flush=True)
                except Exception as e:  # noqa: BLE001
                    import traceback as _tb
                    if model_mutated:
                        print(f"[grpo] resume.pt: model was partially "
                              f"loaded then failed: {type(e).__name__}: {e}"
                              f"\n{_tb.format_exc()}\n  → aborting "
                              f"(refusing to fake a fresh start with "
                              f"half-resumed weights).", flush=True)
                        raise
                    print(f"[grpo] resume.pt: pre-mutation failure: "
                          f"{type(e).__name__}: {e}\n  → starting fresh "
                          f"(model still pristine).", flush=True)
            # PHASE 3: history.json is OPTIONAL. A corrupt history doesn't
            # justify resetting start_step (would let new ckpts overwrite
            # surviving ones). Empty list + correct start_step is fine.
            hist_path = out_path / "history.json"
            if hist_path.exists():
                try:
                    history = _json.loads(hist_path.read_text())
                except _json.JSONDecodeError as e:
                    print(f"[grpo] WARN history.json corrupt "
                          f"({type(e).__name__}: {e}); starting with empty "
                          f"history at step={start_step}.", flush=True)
                    history = []
            print(f"[grpo] RESUMING from {out_path / ck['ckpt_dir_name']} "
                  f"→ start step={start_step} (history has {len(history)} "
                  f"prior steps, "
                  f"elapsed_baseline={resumed_elapsed_baseline/3600:.1f}h)",
                  flush=True)
    else:
        print(f"[grpo] no resume.pt; starting fresh", flush=True)

    # Open a JSONL stream for rich per-step details (per-task, per-arch records).
    # APPEND on resume (so prior step records survive); TRUNCATE on fresh start.
    details_path = out_path / "details.jsonl"
    details_fp = open(details_path, "a" if start_step > 0 else "w", buffering=1)

    # Seeded RNG so inject sampling is reproducible across reruns.
    inj_rng = _random.Random(inject_seed) if inject_seed is not None else None
    if inj_rng is not None and resumed_inj_rng_state is not None:
        # Bit-identical resume of the inject RNG (replaces the old
        # `for _ in range(32*start_step)` approximation).
        try:
            inj_rng.setstate(tuple(resumed_inj_rng_state)
                             if not isinstance(resumed_inj_rng_state, tuple)
                             else resumed_inj_rng_state)
            print(f"[grpo] inj_rng restored from resume.pt (exact)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[grpo] inj_rng restore failed ({e}); approximate "
                  f"fast-forward fallback", flush=True)
            for _ in range(32 * start_step):
                inj_rng.random()

    t0 = time.time()
    try:
        for step, batch in enumerate(batches):
            # Skip already-completed batches on resume
            if step < start_step:
                continue
            step_k = inject_k_per_step[step] if inject_k_per_step is not None else inject_k
            # Step-level resilience: any uncaught exception (OOM, NaN in
            # backward, optimizer crash, etc.) should skip this step and
            # let training continue. The next batch gets a fresh shot.
            try:
                out = grpo_step(model, tokenizer, batch, executor, spec,
                                device=device,
                                max_concurrent_runs=max_concurrent_runs,
                                inject_archs=inject_archs,
                                inject_pool=inject_pool,
                                inject_k=step_k,
                                inject_family_stratified=inject_family_stratified,
                                inject_rng=inj_rng,
                                reward_fn=reward_fn,
                                arch_cache=arch_cache,
                                arch_cache_step=step)
                loss = out["loss"]
                # Guard 1: NaN/Inf loss never flows into backward.
                if not torch.isfinite(loss).all():
                    print(f"[grpo] step={step} loss not finite "
                          f"({loss.item()}); skipping backward.", flush=True)
                    optim.zero_grad(set_to_none=True)
                    continue
                loss.backward()
                # Guard 2: post-backward NaN/Inf in any parameter's grad.
                # `clip_grad_norm_` does NOT catch this — its internal
                # `total_norm > max_norm` comparison silently returns
                # False on NaN, leaving grads unscaled. Then optim.step
                # consumes the NaN grad and permanently NaN-poisons the
                # parameters; every subsequent loss is NaN and the
                # "skip-backward on NaN-loss" path above stops being
                # useful (the model is already dead). Verified May 2026.
                trainable = [p for p in model.parameters() if p.requires_grad]
                bad_grad = False
                for p in trainable:
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        bad_grad = True
                        break
                if bad_grad:
                    print(f"[grpo] step={step} non-finite grad detected "
                          f"after backward; skipping optim.step to preserve "
                          f"params.", flush=True)
                    optim.zero_grad(set_to_none=True)
                    continue
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
            except Exception as e:  # noqa: BLE001
                import traceback as _tb
                print(f"[grpo] step={step} CRASHED, skipping: "
                      f"{type(e).__name__}: {e}\n{_tb.format_exc()}",
                      flush=True)
                optim.zero_grad(set_to_none=True)
                continue

            # Split scalars (history.json) from rich per-task details (details.jsonl).
            DETAIL_KEYS = {"per_task_details", "correct_count_dist"}
            rec = {}
            for k, v in out.items():
                if k in DETAIL_KEYS:
                    continue
                if isinstance(v, torch.Tensor):
                    rec[k] = float(v.detach().item())
                elif isinstance(v, (int, float)):
                    rec[k] = float(v)
                # Lists like n_unanimous/n_mixed are ints, handled above.
            rec["step"] = step
            # `elapsed` is cumulative wall across ALL resume cycles
            # (= prior runs' accumulated elapsed + this run's). That
            # keeps the history.json elapsed series monotonic — a
            # resume from step 50 at hour 8 will record step 51 at
            # hour 8 + (time since resume), not at hour 0.
            now = time.time()
            rec["elapsed"] = resumed_elapsed_baseline + (now - t0)
            rec["step_wall_seconds"] = (
                rec["elapsed"] - history[-1]["elapsed"] if history else rec["elapsed"]
            )
            # Per-step rollups (consumed by operator wake-up + offline
            # analyzer): search-stub counts and tool-error kinds per step
            # so we don't need to re-parse details.jsonl for a quick view.
            stub_by_tool: dict[str, int] = {}
            tek_total: dict[str, int] = {}
            for task_d in (out.get("per_task_details") or []):
                for s in (task_d.get("samples") or []):
                    for kk, vv in (s.get("search_stub_counts") or {}).items():
                        stub_by_tool[kk] = stub_by_tool.get(kk, 0) + int(vv)
                    for kk, vv in (s.get("tool_error_kinds") or {}).items():
                        tek_total[kk] = tek_total.get(kk, 0) + int(vv)
            rec["search_stub_by_tool"] = stub_by_tool
            rec["tool_error_kinds_agg"] = tek_total
            # Cache hit-rate this step (if cache enabled).
            if arch_cache is not None:
                rec.update(arch_cache.pop_step_stats())
            history.append(rec)

            # Write the rich detail row (one JSON per step) — best-effort.
            # NB: we deliberately don't `except Exception: pass` silently
            # here — write failures are logged so we don't lose insight
            # into why our forensic record is empty.
            import json as _json
            try:
                detail_row = {
                    "step": step,
                    "elapsed": rec["elapsed"],
                    "correct_count_dist": out.get("correct_count_dist"),
                    "n_unanimous": out.get("n_unanimous"),
                    "n_mixed": out.get("n_mixed"),
                    "n_eng_invalid": out.get("n_eng_invalid"),
                    "eng_invalid_rate": out.get("eng_invalid_rate"),
                    "n_archs_with_search_stub": out.get("n_archs_with_search_stub"),
                    "search_stub_total":        out.get("search_stub_total"),
                    "n_archs_with_run_errors":  out.get("n_archs_with_run_errors"),
                    "run_error_kinds":          out.get("run_error_kinds"),
                    "per_task_details": out.get("per_task_details"),
                }
                details_fp.write(_json.dumps(detail_row) + "\n")
            except (OSError, TypeError, ValueError) as e:
                print(f"[grpo] WARN: details.jsonl write failed step={step}: "
                      f"{type(e).__name__}: {e}", flush=True)

            if step % log_every == 0:
                # Termination breakdown: turn-level outcomes (sum across
                # G*B traces). Healthy = submit + skip_explicit; the rest
                # are engineering/protocol problems we monitor.
                t_sub = int(rec.get("term_submit", 0))
                t_skp = int(rec.get("term_skip_explicit", 0))
                t_hc  = int(rec.get("term_skip_hit_cap", 0))
                t_we  = int(rec.get("term_skip_worker_err", 0))
                t_wc  = int(rec.get("term_skip_wall_clock", 0))
                t_emp = int(rec.get("term_skip_empty", 0))
                total_turns = t_sub + t_skp + t_hc + t_we + t_wc + t_emp
                # Attribution:
                #   Architecture-attributable: submit / skip_explicit /
                #     skip_hit_cap / skip_wall_clock (arch's chosen path
                #     overran one of its bounds).
                #   Engineering noise: skip_worker_error / skip_empty_text
                #     (API gateway / model returned blank — infra hiccup).
                eng_noise = t_we + t_emp
                eng_pct = (eng_noise / max(1, total_turns)) * 100
                stub_total = int(rec.get("search_stub_total", 0))
                n_run_err = int(rec.get("n_archs_with_run_errors", 0))
                print(
                    f"[grpo] step={step:>4} "
                    f"L={rec['loss']:.3f} pg={rec['loss_pg']:.3f} "
                    f"H={rec['entropy']:.2f} "
                    f"r̄={rec['reward_mean']:.3f}±{rec['reward_std']:.3f} "
                    f"unanimous={int(rec.get('n_unanimous', 0))}/{int(rec.get('n_unanimous', 0))+int(rec.get('n_mixed', 0))} "
                    f"wall[p90={rec.get('trace_wall_p90', 0):.0f}s max={rec.get('trace_wall_max', 0):.0f}s] "
                    f"turns[submit={t_sub} skip={t_skp} hc={t_hc} wc={t_wc} | werr={t_we} emp={t_emp}] "
                    f"eng_noise={eng_pct:.1f}% "
                    f"search_stub={stub_total} run_err={n_run_err}"
                )
                if wandb_run is not None:
                    wandb_run.log(rec, step=step)

            if (step + 1) % save_every == 0:
                from .sft import save_head_checkpoint
                tag = f"grpo_step{step+1}"
                save_head_checkpoint(model, out_path, tag=tag)
                # Atomic resume pointer: optimizer + last-completed-step.
                # Written via temp-file + rename so a crash mid-write can't
                # leave a corrupt resume.pt (next launch would then start
                # from scratch via the except path).
                try:
                    tmp = resume_pt.with_suffix(".pt.tmp")
                    torch.save({
                        "ckpt_dir_name": f"head_{tag}",
                        "optim_state_dict": optim.state_dict(),
                        "last_completed_step": step,
                        # Resume completeness extras: inject RNG state
                        # (bit-identical reproduction), arch_cache RNG
                        # (same), and an elapsed baseline so wall-clock
                        # metrics in history.json stay monotonic across
                        # resume cycles.
                        "inj_rng_state": (inj_rng.getstate()
                                          if inj_rng is not None else None),
                        "cache_rng_state": (arch_cache.rng_state()
                                            if arch_cache is not None else None),
                        "elapsed_baseline_s": float(
                            resumed_elapsed_baseline + (time.time() - t0)
                        ),
                    }, tmp)
                    tmp.replace(resume_pt)
                except Exception as e:  # noqa: BLE001
                    print(f"[grpo] resume.pt save failed (training continues): "
                          f"{type(e).__name__}: {e}", flush=True)
            # Dump history.json atomically (tmp + rename) after each
            # step so a crash mid-write can't leave a half-truncated
            # JSON that resume's `json.loads` would choke on (BUG #2).
            try:
                import json as _json
                hist_path = out_path / "history.json"
                tmp = hist_path.with_suffix(".json.tmp")
                with open(tmp, "w") as _f:
                    _json.dump(history, _f, indent=2)
                tmp.replace(hist_path)
            except Exception as e:  # noqa: BLE001
                print(f"[grpo] WARN history.json write failed "
                      f"step={step}: {type(e).__name__}: {e}", flush=True)
    finally:
        details_fp.close()

    from .sft import save_head_checkpoint
    save_head_checkpoint(model, out_path, tag="grpo_final")

    # `final_step` is the highest step counter in history (matches the
    # SFT convention); using `len(history)` would mis-report progress
    # after a resume that started with empty history.
    final_step = max((r.get("step", -1) for r in history), default=-1) + 1
    return {"history": history, "final_step": final_step}


__all__ = [
    "DEFAULT_ENTROPY_WEIGHTS",
    "GRPOBatch",
    "entropy_typed",
    "grpo_step",
    "shaped_advantage",
    "train_grpo",
]
