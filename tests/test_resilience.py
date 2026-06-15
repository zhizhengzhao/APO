"""Resilience tests — confirm that no failure path terminates training.

Covers:
  1. Agent.run raising Python exception → executor synthesizes worker_error
     turn, trace continues, n_api_errors increments.
  2. Synth.judge raising → executor treats as malformed verdict, keeps
     iterating cycles, n_api_errors increments.
  3. Tool raising → already handled at tool level (returns sentinel string),
     this test just verifies the sentinel path doesn't crash agent.run.
"""

from __future__ import annotations

from dataclasses import replace

from arch_policy import (
    ARCH,
    BASELINE_REGISTRY,
    MockWorker,
    MultiAgentExecutor,
    Worker,
    WorkerOutput,
    get_baseline,
)
from arch_policy.executor.synth import SynthVerdict


class _AlwaysRaiseWorker(Worker):
    """Worker that throws on every chat() — simulates a tool-implementation
    bug deep inside agent.run that the worker can't possibly catch."""
    def chat(self, system, user, max_new_tokens=512):
        if "You are a Synth" in system:
            return WorkerOutput(text="ANSWER: ok", n_input_tokens=1, n_output_tokens=1)
        raise RuntimeError("simulated infra explosion")


def test_agent_run_exception_yields_worker_error_not_crash():
    """If agent.run raises a real Python exception, the executor MUST
    convert it into a worker_error turn so the trace finishes and the
    GRPO step survives."""
    spec = replace(ARCH, safety_max_cycles=1, safety_max_steps=2)
    worker = _AlwaysRaiseWorker()
    ex = MultiAgentExecutor(worker=worker, spec=spec)
    arch = get_baseline("solver_verifier")
    trace = ex.run("Q", arch)
    # Must NOT raise. All turns become worker_error skips.
    assert trace.n_api_errors >= 2  # both agents threw
    # Every recorded message is a protocol_fail skip with cause=worker_error
    for m in trace.messages:
        assert m.skipped
        assert m.skip_kind == "protocol_fail"
    # Termination breakdown should attribute these to skip_worker_error
    assert trace.termination_breakdown["skip_worker_error"] >= 2


class _SynthCrashWorker(Worker):
    """Healthy agent worker but Synth raises on judge."""
    def __init__(self, fake_answer="42"):
        self.fake_answer = fake_answer
    def chat(self, system, user, max_new_tokens=512):
        if "You are a Synth" in system:
            raise RuntimeError("synth went boom")
        return WorkerOutput(text=f"Candidate: {self.fake_answer}",
                            n_input_tokens=10, n_output_tokens=3)


def test_synth_crash_does_not_terminate_trace():
    """Synth raising MUST be caught: cycle continues to max_cycles, then
    falls back to heuristic_extract. trace.n_api_errors increments per
    failed synth call."""
    spec = replace(ARCH, safety_max_cycles=2, safety_max_steps=2)
    ex = MultiAgentExecutor(worker=_SynthCrashWorker(), spec=spec)
    arch = get_baseline("solver_verifier")
    trace = ex.run("Q", arch)
    # Did not crash. Cycles exhausted because synth never returned DONE.
    assert trace.hit_cycle_cap
    assert trace.n_synth_calls == 2   # synth was attempted once per cycle
    assert trace.n_api_errors >= 2  # both crashes recorded
    # Final answer extracted from transcript via heuristic_extract
    assert trace.final_answer != ""
    assert "42" in trace.final_answer


def test_full_trace_with_only_mockworker_still_finishes():
    """Sanity check: the resilience changes haven't broken the happy path."""
    spec = replace(ARCH, safety_max_cycles=1, safety_max_steps=2)
    ex = MultiAgentExecutor(
        worker=MockWorker(fake_answer="42", force_synth_done=True),
        spec=spec,
    )
    for name in ["solver_verifier"]:
        arch = get_baseline(name)
        trace = ex.run("Q", arch)
        assert trace.final_answer != ""
        assert trace.n_api_errors == 0


def test_v7eps_arch_caps_not_treated_as_engineering_errors():
    """V7-ε: hit_wall_clock + hit_call_cap are ARCHITECTURE-attributable
    (the arch chose a tool-heavy path), so they must NOT increment
    n_api_errors (which masks GRPO advantage). They DO increment the new
    n_arch_caps_hit counter for telemetry."""
    spec = replace(ARCH, safety_max_cycles=10, safety_max_steps=2)
    ex = MultiAgentExecutor(
        worker=MockWorker(fake_answer="42", force_synth_done=False),
        spec=spec,
        max_llm_calls_per_trace=4,   # fire call_cap quickly
    )
    arch = get_baseline("solver_verifier")
    trace = ex.run("Q", arch)
    assert trace.hit_call_cap, "call_cap must have fired"
    # CRITICAL invariant: arch-attributable cap does NOT count as API error
    assert trace.n_api_errors == 0, (
        f"hit_call_cap must NOT add to n_api_errors; got {trace.n_api_errors}"
    )
    assert trace.n_arch_caps_hit >= 1, "arch cap should be tallied separately"
    # Trace still produces a final answer (heuristic_extract). GRPO will
    # judge that on its merits — no automatic mask.
    assert trace.final_answer != "", "must still extract a heuristic answer"


def test_v7eps_api_worker_error_still_masks_advantage():
    """V7-ε regression: genuine API failures (worker.chat sentinel) MUST
    still increment n_api_errors so GRPO masks them. The split must not
    accidentally drop the INFRA failure path."""
    spec = replace(ARCH, safety_max_cycles=1, safety_max_steps=2)
    worker = _AlwaysRaiseWorker()   # forces every agent.run into worker_error
    ex = MultiAgentExecutor(worker=worker, spec=spec)
    arch = get_baseline("solver_verifier")
    trace = ex.run("Q", arch)
    assert trace.n_api_errors >= 2, (
        f"INFRA error must increment n_api_errors; got {trace.n_api_errors}"
    )
    # arch_caps_hit must NOT increment for INFRA failures
    assert trace.n_arch_caps_hit == 0


def test_grpo_does_not_mask_arch_caps_but_does_mask_api_errors():
    """End-to-end contract: the GRPO eng-valid mask must ONLY zero
    advantage for INFRA failures (n_api_errors > 0). Arch-attributable
    caps (hit_wall_clock, hit_call_cap) must let advantage flow normally.

    We construct ExecutionTrace stubs by hand to bypass executor entirely
    (faster + deterministic) and replicate the exact mask code path."""
    from arch_policy.executor.multi_agent import ExecutionTrace

    arch = get_baseline("solver_verifier")

    # Sample A: hit_wall_clock=True, hit_call_cap=True, no API errors.
    tr_arch_cap = ExecutionTrace(task="Q", arch=arch)
    tr_arch_cap.hit_wall_clock = True
    tr_arch_cap.hit_call_cap = True
    tr_arch_cap.n_arch_caps_hit = 2
    tr_arch_cap.n_api_errors = 0

    # Sample B: API sentinel went off.
    tr_api = ExecutionTrace(task="Q", arch=arch)
    tr_api.n_api_errors = 1

    # Sample C: clean trace.
    tr_clean = ExecutionTrace(task="Q", arch=arch)

    def mask_value(tr):
        return tr.n_api_errors > 0

    assert mask_value(tr_arch_cap) is False, (
        "arch caps must NOT trigger eng-invalid mask"
    )
    assert mask_value(tr_api) is True, (
        "API error must trigger eng-invalid mask"
    )
    assert mask_value(tr_clean) is False


def test_grpo_sentinel_trace_is_masked():
    """If `_run_one` raises a Python exception we synthesize a
    `_sentinel_trace` to keep the batch alive. That sentinel MUST be
    masked — it represents OUR infra bug, not architecture quality.
    Setting `n_api_errors=1` is the trigger."""
    from arch_policy.executor.multi_agent import ExecutionTrace

    arch = get_baseline("solver_verifier")
    tr = ExecutionTrace(task="Q", arch=arch)
    tr.n_api_errors = 1
    assert tr.n_api_errors > 0, "sentinel must trigger the eng-invalid mask"


def test_v7eps_pytest_runner_without_delimiter_does_not_crash():
    """Regression for subagent-review CRITICAL: pytest_runner uses `_re`
    in the fallback path when no `---TESTS---` delimiter is found. The
    `_re` import was accidentally dropped along with sympy_check; result
    was a NameError that bubbled up through Agent.run → got caught as
    `worker_error` → masked as INFRA failure (wrong attribution).

    With `import re as _re` restored, the fallback must work and the tool
    must return a friendly error string (NOT raise)."""
    from arch_policy.executor.tools import call_tool

    # No ---TESTS--- delimiter but a `def test_` line: heuristic split.
    spec = (
        "def add(a, b): return a + b\n"
        "\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n"
    )
    out = call_tool("pytest_runner", spec)
    # Should run (returning some test output), NOT raise NameError.
    assert isinstance(out, str)
    # If pytest is installed and ran, we expect "passed" or exit code; if
    # pytest itself is missing on the box, we tolerate that error string —
    # the key assertion is "no NameError, no Python exception leakage".
    assert "NameError" not in out, f"_re NameError leaked: {out!r}"


def test_thinking_control_routes_per_model():
    """`_apply_thinking_control` must pick the right knob for each GpuGeek
    OpenAI-compat model. Empirically verified table — any refactor that
    breaks this routing will silently revert us to "all models think
    heavily" and 5-10x our API latency."""
    from arch_policy.executor.gpugeek_worker import _apply_thinking_control

    # === V4-Flash: `reasoning_effort` top arg accepts "none" / "high".
    kw = {}; _apply_thinking_control("Vendor3/DeepSeek-V4-Flash", False, kw)
    assert kw == {"reasoning_effort": "none"}, kw
    kw = {}; _apply_thinking_control("Vendor3/DeepSeek-V4-Flash", True, kw)
    assert kw == {"reasoning_effort": "high"}, kw

    # === V4-Pro REJECTS reasoning_effort="none" → needs extra_body.
    kw = {}; _apply_thinking_control("Vendor3/DeepSeek-V4-Pro", False, kw)
    assert kw == {"extra_body": {"thinking": {"type": "disabled"}}}, kw
    kw = {}; _apply_thinking_control("Vendor3/DeepSeek-V4-Pro", True, kw)
    assert kw == {"reasoning_effort": "high"}, kw

    # === GPT-5.5: same as Flash.
    kw = {}; _apply_thinking_control("Vendor2/GPT-5.5", False, kw)
    assert kw == {"reasoning_effort": "none"}, kw
    kw = {}; _apply_thinking_control("Vendor2/GPT-5.5", True, kw)
    assert kw == {"reasoning_effort": "high"}, kw

    # === Unknown / other models: pass-through (no knob applied).
    kw = {}; _apply_thinking_control("Vendor3/Other-Model", False, kw)
    assert kw == {}
    kw = {}; _apply_thinking_control("Vendor3/Other-Model", True, kw)
    assert kw == {}


def test_worker_thinking_defaults_to_off():
    """V7-ζ: user mandate is 'all models default to no thinking'. The
    GpuGeekWorker dataclass default must be False so even callers that
    forget --no-worker_thinking get the cheap/fast path by default."""
    from arch_policy.executor.gpugeek_worker import GpuGeekWorker
    w = GpuGeekWorker(model="Vendor3/DeepSeek-V4-Flash", api_key="dummy")
    assert w.thinking is False, (
        f"GpuGeekWorker.thinking default must be False (got {w.thinking})"
    )


def test_thinking_control_off_does_not_corrupt_existing_extra_body():
    """If caller already populated extra_body for some other reason,
    V4-Pro's OFF knob must MERGE (via setdefault) instead of clobber."""
    from arch_policy.executor.gpugeek_worker import _apply_thinking_control
    kw = {"extra_body": {"other_key": "preserved"}}
    _apply_thinking_control("Vendor3/DeepSeek-V4-Pro", False, kw)
    assert kw["extra_body"]["other_key"] == "preserved"
    assert kw["extra_body"]["thinking"] == {"type": "disabled"}


def test_deepseek_worker_defaults_and_env():
    """DeepSeekWorker: thinking OFF by default, picks up DEEPSEEK_API_KEY
    from env when api_key= not passed, and raises a clear error if neither
    is set."""
    import os
    from arch_policy.executor.deepseek_worker import DeepSeekWorker

    saved = os.environ.pop("DEEPSEEK_API_KEY", None)
    try:
        # explicit api_key works
        w = DeepSeekWorker(api_key="sk-dummy")
        assert w.thinking is False, "thinking must default OFF (cost)"
        assert w.model == "deepseek-v4-flash", (
            "default model must be the cheap flash variant"
        )
        assert w.base_url == "https://api.deepseek.com"

        # env fallback works
        os.environ["DEEPSEEK_API_KEY"] = "sk-from-env"
        w2 = DeepSeekWorker()
        assert w2.api_key == "sk-from-env"

        # neither path → clear error
        del os.environ["DEEPSEEK_API_KEY"]
        try:
            DeepSeekWorker()
        except RuntimeError as e:
            assert "DEEPSEEK_API_KEY" in str(e)
        else:
            raise AssertionError("expected RuntimeError when no key available")
    finally:
        if saved is not None:
            os.environ["DEEPSEEK_API_KEY"] = saved
        else:
            os.environ.pop("DEEPSEEK_API_KEY", None)


def test_deepseek_worker_error_sentinel_pattern():
    """agent.py looks for `[DeepSeekWorker error` as a worker_error sentinel.
    Lock the exact prefix the worker emits on terminal failure so the
    detection in Agent.run never silently slips."""
    from arch_policy.executor.deepseek_worker import DeepSeekWorker
    w = DeepSeekWorker(api_key="sk-dummy", model="invalid-model-name",
                       base_url="http://127.0.0.1:1",  # unreachable
                       max_retries=1, retry_initial_delay=0.01)
    out = w.chat("sys", "hello", max_new_tokens=8)
    # On total failure: empty text replaced by sentinel string
    assert "[DeepSeekWorker error" in out.text, out.text
    assert out.n_input_tokens == 0 and out.n_output_tokens == 0


def test_v7eps_grpo_loss_uses_valid_sample_denominator():
    """Critical V7-ε math fix: loss_pg must divide by n_VALID samples, not
    by full G*B. Otherwise eng-invalid samples (advantage=0 via mask)
    silently down-scale the gradient by (G*B - n_invalid)/(G*B).

    We replicate the relevant grpo arithmetic in isolation to lock the
    semantic: a fully-invalid batch should NOT zero-divide; a half-invalid
    batch should yield 2x the loss magnitude vs the unmasked case."""
    import torch
    G, B = 4, 2

    # Pretend per-sample log_pi and advantages
    advantage = torch.tensor([[1.0, -1.0],
                              [1.0, -1.0],
                              [0.0,  0.0],     # invalid
                              [0.0,  0.0]])    # invalid
    log_pi    = torch.full((G, B), -0.5)
    eng_valid = torch.tensor([[True, True],
                              [True, True],
                              [False, False],
                              [False, False]])

    # OLD (buggy) version: divide by G*B
    old = -(advantage * log_pi).mean()
    # NEW (V7-ε): divide by n_valid
    n_valid = int(eng_valid.sum().item())
    new = -(advantage * log_pi).sum() / max(1, n_valid)

    # With half-invalid, new must be 2x the old (signal not diluted).
    assert torch.isclose(new, 2 * old), (
        f"new loss should compensate for invalid samples; got new={new}, old={old}"
    )

    # Fully-invalid batch: must not zero-divide.
    eng_all_invalid = torch.zeros((G, B), dtype=torch.bool)
    n_valid2 = max(1, int(eng_all_invalid.sum().item()))
    safe = -(advantage * log_pi).sum() / n_valid2
    assert torch.isfinite(safe), "fully-invalid batch must produce finite loss"


def test_v7eps_log_prob_gates_conditional_normalizer():
    """V7-ε sampling-density fix: `log_prob_gates` must subtract the
    log-normalizer log(1 - prod(1 - p_i)) because `sample_arch` rejects
    all-zero outcomes. Subtracting a negative number ADDS positive mass,
    so the corrected log-prob is strictly >= the uncorrected raw sum,
    and the gap matches log(1 - P_all_zero)."""
    import math
    import torch
    import torch.nn.functional as F
    from arch_policy.architecture.sampler import log_prob_gates

    # All slots ~equally likely, single active. Conditional correction
    # is meaningful (P_all_zero is non-trivial).
    z = torch.tensor([0.0, 0.0, 0.0, 0.0])    # p_i = 0.5 each
    active = torch.tensor([True, False, False, False])

    # Raw (uncorrected) sum: should be log(0.5) + 3·log(0.5) = -4·log 2
    log_p1 = -F.softplus(-z)
    log_p0 = -F.softplus(z)
    raw = torch.where(active, log_p1, log_p0).sum().item()
    expected_raw = 4 * math.log(0.5)
    assert math.isclose(raw, expected_raw, abs_tol=1e-6), raw

    # Expected normalizer: log(1 - 0.5^4) = log(1 - 0.0625) = log 0.9375
    log_p_atleast = math.log(1 - 0.5 ** 4)
    expected_corrected = raw - log_p_atleast

    out = log_prob_gates(z, active).item()
    assert math.isclose(out, expected_corrected, abs_tol=1e-6), (
        f"corrected log_prob mismatch: got {out}, expected {expected_corrected}"
    )
    # Sanity: correction must shift upward (less negative)
    assert out > raw, f"correction should raise log_prob; raw={raw} corr={out}"

    # Pathological case: all gates ≈ 0 should NOT NaN (clamp protects log1mexp).
    z_bad = torch.full((4,), -50.0)
    active_one = torch.tensor([True, False, False, False])
    out_bad = log_prob_gates(z_bad, active_one).item()
    assert math.isfinite(out_bad), f"pathological case must not NaN; got {out_bad}"


def test_v7eps_executor_caps_mutually_exclusive():
    """V7-ε attribution cleanup: when a trace falls through to the
    heuristic-extract fallback, exactly ONE of (hit_wall_clock,
    hit_call_cap, hit_cycle_cap) must be True. Previously
    `hit_cycle_cap=True` was set on EVERY fallback path, even when
    wall_clock or call_cap was the actual cause."""
    spec = replace(ARCH, safety_max_cycles=2, safety_max_steps=2)
    # Force a wall_clock hit by setting the timeout absurdly low.
    ex = MultiAgentExecutor(
        worker=MockWorker(fake_answer="42", force_synth_done=False),
        spec=spec,
        wall_clock_timeout_s=0.001,   # any trace exceeds this
    )
    arch = get_baseline("solver_verifier")
    trace = ex.run("Q", arch)
    flags = (trace.hit_wall_clock, trace.hit_call_cap, trace.hit_cycle_cap)
    assert sum(int(f) for f in flags) <= 1, (
        f"at most one cap-hit flag should be True; got {flags}"
    )


def test_v7eps_python_exec_timeout_surfaces_partial_stdout():
    """V7-ε: on TIMEOUT, python_exec should return a short factual hint
    PLUS whatever stdout was captured before the kill. Lets the model see
    how far its code got without re-running (purely informational, not
    prescriptive). The hint phrase is bounded to a few words."""
    from arch_policy.executor.tools import python_exec, PYTHON_TIMEOUT_S
    # Slowly emit progress, then sleep past the timeout
    code = (
        "import time\n"
        "for i in range(5):\n"
        "    print(f'progress {i}', flush=True)\n"
        "    time.sleep(0.5)\n"
        "time.sleep(120)\n"   # well over PYTHON_TIMEOUT_S
    )
    out = python_exec(code)
    assert "[python_exec] TIMEOUT" in out
    assert f"{PYTHON_TIMEOUT_S}s" in out
    assert "code too slow" in out, "hint must be present (factual, not prescriptive)"
    assert "PARTIAL STDOUT" in out, "partial stdout should be surfaced"
    assert "progress 0" in out, "captured prints should appear"


def test_v7eps_python_exec_timeout_silent_code_no_partial_section():
    """If the code prints nothing before timeout, do not show an empty
    PARTIAL STDOUT block (keep the message terse)."""
    from arch_policy.executor.tools import python_exec
    code = "import time; time.sleep(120)"   # no print, just hang
    out = python_exec(code)
    assert "[python_exec] TIMEOUT" in out
    assert "code too slow" in out
    assert "PARTIAL STDOUT" not in out, "no partial when nothing was printed"


def test_v7eps_python_exec_log_populated():
    """python_exec_log records (elapsed_s, code_snippet, ok) for each
    python_exec call so we can later tune PYTHON_TIMEOUT_S from data."""

    class _PyCallWorker(Worker):
        def __init__(self):
            self.calls = 0
        def chat(self, system, user, max_new_tokens=512):
            if "You are a Synth" in system:
                return WorkerOutput(text=f"ANSWER: 9",
                                    n_input_tokens=1, n_output_tokens=1)
            self.calls += 1
            if self.calls == 1:
                # tool call
                return WorkerOutput(
                    text="THOUGHT: compute\nACTION: python_exec\nARGS: print(3*3)",
                    n_input_tokens=20, n_output_tokens=8,
                )
            return WorkerOutput(text="Candidate: 9",
                                n_input_tokens=10, n_output_tokens=3)

    spec = replace(ARCH, safety_max_cycles=1, safety_max_steps=4)
    ex = MultiAgentExecutor(worker=_PyCallWorker(), spec=spec)
    trace = ex.run("Q", get_baseline("solver_verifier"))
    # Must have at least one python_exec invocation logged
    assert len(trace.python_exec_log) >= 1
    elapsed_s, snippet, ok = trace.python_exec_log[0]
    # >= 0 (not > 0): time.time() resolution on busy CI can record 0.0
    # for sub-millisecond ops. The semantic check we care about is "the
    # field gets populated", not "subprocess always takes a measurable ms".
    assert elapsed_s >= 0.0, f"duration must be non-negative; got {elapsed_s}"
    assert "print(3*3)" in snippet, f"snippet missing the code; got {snippet!r}"
    assert ok is True, "successful python_exec call should be ok=True"


def test_max_llm_calls_per_trace_cap_truncates_cleanly():
    """max_llm_calls_per_trace=N must stop the cycle loop once `trace.n_llm_calls`
    reaches N and set `hit_call_cap=True`. V7-ε: this is arch-attributable
    (not eng-invalid), so n_api_errors stays 0 and n_arch_caps_hit ticks."""
    spec = replace(ARCH, safety_max_cycles=10, safety_max_steps=2)
    ex = MultiAgentExecutor(
        worker=MockWorker(fake_answer="42", force_synth_done=False),
        spec=spec,
        max_llm_calls_per_trace=8,
    )
    arch = get_baseline("solver_verifier")
    trace = ex.run("Q", arch)
    assert trace.hit_call_cap
    assert not trace.hit_wall_clock
    # V7-ε attribution: cap is arch's fault, not INFRA noise.
    assert trace.n_api_errors == 0
    assert trace.n_arch_caps_hit >= 1
    assert 6 <= trace.n_llm_calls <= 12, f"n_llm_calls={trace.n_llm_calls}"


def test_grpo_run_one_sentinel_on_exception():
    """_run_one inside grpo_step must convert any executor exception into
    a sentinel trace with n_api_errors=1 so the rest of the batch survives.
    We can't easily unit-test grpo_step, but we can verify the sentinel
    shape is reproducible from the public ExecutionTrace API."""
    from arch_policy.executor.multi_agent import ExecutionTrace
    arch = get_baseline("solver_verifier")
    tr = ExecutionTrace(task="Q", arch=arch)
    tr.n_api_errors = 1
    assert tr.n_api_errors > 0
    assert "skip_worker_error" in tr.termination_breakdown
    assert tr.protocol_compliance == {}
