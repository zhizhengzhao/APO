"""Pure-data classes for executor telemetry.

`AgentMessage`     — one agent's reply at one (cycle, turn) position.
`ExecutionTrace`   — everything we record about a single run() call.

These were split out of `multi_agent.py` so the executor file stays
focused on control flow; downstream code (analyzer, tests, training
loop) imports the dataclasses from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..architecture.sampler import ConcreteArch


@dataclass
class AgentMessage:
    """One agent's reply (or skip marker) at one (cycle, turn) position.

    `skipped=True` covers both explicit skips and protocol failures;
    `text == ""` in both cases. Downstream agents' incoming filter drops
    skipped entries. Synth sees the full transcript and renders empty
    messages as a `[empty / skipped]` placeholder (see
    `prompts.format_full_transcript`) so the judge can tell agents
    declined to contribute. R37's mask-softening correctness rests on
    this rendering: synth verdicts produced from a transcript
    containing only placeholder entries (no real content) are
    expected to fail-closed, but mixed placeholder + real content is
    trustworthy signal about the surviving agents' answer.
    """
    slot: int
    role: str
    cycle: int
    turn: int
    text: str
    n_steps: int = 0
    n_tool_calls: int = 0
    n_real_tool_calls: int = 0    # == n_tool_calls (submit/skip don't dispatch)
    n_input_tokens: int = 0
    n_output_tokens: int = 0
    hit_step_cap: bool = False
    skipped: bool = False
    skip_kind: str = ""           # "" | "explicit" | "protocol_fail"


@dataclass
class ExecutionTrace:
    """Everything one MultiAgentExecutor.run() call records.

    Field groups:
      identity         : task, arch
      transcript       : messages, final_answer, synth_log
      counts           : n_llm_calls, n_cycles_run, n_synth_calls, tokens
      cap-hit flags    : hit_cycle_cap / hit_wall_clock / hit_call_cap (mutually
                         exclusive; arch-attributable)
      tool telemetry   : tool_call_counts / tool_error_counts / tool_error_kinds
                         + python_exec_log (per-call duration for cap tuning)
      eng vs arch      : n_api_errors + n_worker_truncations (infra,
                         gradient-masked) vs n_arch_caps_hit (arch's
                         choice, gradient kept). Split so dashboard
                         distinguishes real API failures from model
                         verbosity hitting max_new_tokens.
      protocol         : protocol_compliance per role (skipped vs submitted),
                         termination_breakdown (7-way per-turn outcome)
      silent-degradation watch:
        search_stub_counts — non-zero means Serper returned stubs (key
        missing or HTTP error); training scripts run preflight to prevent
        this in production, but the field stays so degraded ad-hoc runs
        are detectable in post-mortem.
        run_errors — structured {kind,type,message,traceback} so we
        never lose forensic detail when grpo._run_one catches an
        exception.
    """
    task: str
    arch: ConcreteArch
    messages: list[AgentMessage] = field(default_factory=list)
    final_answer: str = ""
    n_llm_calls: int = 0
    n_cycles_run: int = 0
    n_synth_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    wall_seconds: float = 0.0   # measured run() wall-clock for this trace
    synth_log: list[str] = field(default_factory=list)
    final_via_synth: bool = False
    # Cap-hit flags (mutually exclusive)
    hit_cycle_cap: bool = False
    hit_wall_clock: bool = False
    hit_call_cap: bool = False
    n_agents_hit_step_cap: int = 0
    # Tool telemetry
    total_tool_calls: int = 0
    tool_call_counts: dict = field(default_factory=dict)
    tool_error_counts: dict = field(default_factory=dict)
    tool_error_kinds: dict = field(default_factory=dict)   # "<tool>:<kind>" → n
    # First 800 chars of each python_exec call's code so forensic review
    # can see what timed out (not just imports). Each entry:
    # (elapsed_s, code_snippet, ok).
    python_exec_log: list[tuple[float, str, bool]] = field(default_factory=list)
    n_tool_truncations: int = 0
    # Engineering vs architecture failure separation. BOTH n_api_errors
    # and n_worker_truncations are infra-attributable (arch's choice can't
    # affect API gateway state nor the max_new_tokens cap); GRPO's
    # eng_valid mask treats either > 0 as engineering noise.
    #
    # We split them so the dashboard distinguishes:
    #   - n_api_errors: worker chat exhausted its 6-retry budget (real
    #     network / 5xx / API gateway issue). Watchdog `api_error_high`
    #     fires on this — that signal SHOULD wake an operator.
    #   - n_worker_truncations: worker output hit max_new_tokens cap
    #     (finish_reason='length' before submit/skip). Verbose LLM
    #     behavior, not infra degradation. Useful for max_new_tokens
    #     tuning but routine.
    # Pre-split (May 2026): both events bumped n_api_errors. Live data
    # showed 181/181 "api errors" were actually truncations, masking a
    # real network failure spike would have been impossible to spot.
    n_api_errors:         int = 0   # 6-retry exhausted (true infra fail)
    n_worker_truncations: int = 0   # output cap hit (model verbosity)
    n_arch_caps_hit:      int = 0   # arch chose a path that ran over; reward kept
    # Protocol compliance per role: {role: {submit_implicit, skipped_explicit,
    # skipped_protocol_fail, zero_real_tool_submits}}.
    protocol_compliance: dict = field(default_factory=dict)
    n_skipped_turns: int = 0
    n_protocol_fail_turns: int = 0
    # Per-turn termination outcome (7-way; healthy = submit + skip_explicit).
    # Keys MUST exactly cover every `skip_<cause>` that multi_agent.py
    # builds from AgentTurnOutput.skip_cause — see the docstring on
    # AgentTurnOutput.skip_cause for the canonical list. A missing key
    # silently mis-attributes (e.g. truncated → empty_text), poisoning
    # the protocol-compliance breakdown analyzers read.
    termination_breakdown: dict = field(default_factory=lambda: {
        "submit_implicit":   0,
        "skipped_explicit":  0,
        "skip_hit_cap":      0,
        "skip_worker_error": 0,
        "skip_wall_clock":   0,
        "skip_truncated":    0,
        "skip_empty_text":   0,
    })
    # Silent-degradation watch (search-tool offline stubs per trace).
    # Keys MUST stay in sync with `tools.SEARCH_TOOL_NAMES` — kept as a
    # literal here to avoid a `tools → trace` circular import. New
    # tools that route through `_serper_post` (or otherwise call
    # `_stub_counts`) need a key added on both sides.
    search_stub_counts: dict = field(default_factory=lambda: {
        "web_search": 0, "arxiv_search": 0, "wikipedia_search": 0,
    })
    # Structured run-level errors: {kind, type, message, traceback?}.
    run_errors: list[dict] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


__all__ = ["AgentMessage", "ExecutionTrace"]
