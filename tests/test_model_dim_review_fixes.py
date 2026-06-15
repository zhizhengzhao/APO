"""Regression tests for the multi-model review batch (2026-05-29).

C1  arch_cache key includes model (multi-model archs don't collide;
    single-model key byte-identical → existing cache stays valid).
C2  per-model ConcurrencyLimitedWorker caps in-flight calls.
H1  LiveCodeBench functional (call-method) grading, not just stdin.
M1  qwen worker treats 429/rate-limit as retriable even if the message
    also contains "quota".
(H2 active_models + C3 eval multi-model are exercised by inspection /
 the existing grpo + eval paths; C1/C2/H1/M1 get focused unit tests.)
"""

from __future__ import annotations

import torch

from arch_policy.architecture.sampler import ConcreteArch


def _arch(model=None):
    return ConcreteArch(
        active_mask=torch.tensor([True, True] + [False] * 4),
        roles=torch.tensor([2, 4] + [0] * 4, dtype=torch.long),
        edges=torch.zeros(6, 6, dtype=torch.bool),
        sequence=torch.tensor([0, 1], dtype=torch.long),
        model=model,
    )


# ---------------------------------------------------------------------------
# C1 — arch_cache key includes model
# ---------------------------------------------------------------------------

def test_C1_single_model_key_unchanged():
    """model=None must hash identically to the pre-model-dim version so
    an existing arch_cache stays valid on resume."""
    from arch_policy.training.arch_cache import arch_key
    import hashlib
    a = _arch(model=None)
    # recompute the OLD 4-field hash explicitly
    blob = b"".join((
        a.active_mask.cpu().to(torch.bool).contiguous().numpy().tobytes(),
        a.roles.cpu().to(torch.long).contiguous().numpy().tobytes(),
        a.edges.cpu().to(torch.bool).contiguous().numpy().tobytes(),
        a.sequence.cpu().to(torch.long).contiguous().numpy().tobytes(),
    ))
    old = hashlib.sha1(blob).hexdigest()[:16]
    assert arch_key(a) == old, "single-model key drifted → would invalidate cache"


def test_C1_model_assignment_changes_key():
    """Two archs identical except model assignment MUST hash differently,
    else the cache returns the wrong model's reward."""
    from arch_policy.training.arch_cache import arch_key
    a1 = _arch(model=torch.tensor([0, 1] + [0] * 4, dtype=torch.long))
    a2 = _arch(model=torch.tensor([1, 0] + [0] * 4, dtype=torch.long))
    assert arch_key(a1) != arch_key(a2)
    # and same model → same key
    a3 = _arch(model=torch.tensor([0, 1] + [0] * 4, dtype=torch.long))
    assert arch_key(a1) == arch_key(a3)


def test_C1_multi_model_differs_from_single():
    from arch_policy.training.arch_cache import arch_key
    a_none = _arch(model=None)
    a_model = _arch(model=torch.tensor([0, 0] + [0] * 4, dtype=torch.long))
    # even all-zeros model assignment differs from None (None = no dim)
    assert arch_key(a_none) != arch_key(a_model)


# ---------------------------------------------------------------------------
# C2 — per-model concurrency semaphore
# ---------------------------------------------------------------------------

def test_C2_concurrency_limited_worker_caps_inflight():
    import threading, time
    from arch_policy.executor.multi_agent import (
        ConcurrencyLimitedWorker, Worker, WorkerOutput)

    peak = {"now": 0, "max": 0}
    lock = threading.Lock()

    class _SlowWorker(Worker):
        def chat(self, system, user, max_new_tokens=512):
            with lock:
                peak["now"] += 1
                peak["max"] = max(peak["max"], peak["now"])
            time.sleep(0.05)
            with lock:
                peak["now"] -= 1
            return WorkerOutput(text="ok", n_input_tokens=1, n_output_tokens=1)

    w = ConcurrencyLimitedWorker(_SlowWorker(), max_concurrency=3)
    threads = [threading.Thread(target=lambda: w.chat("s", "u")) for _ in range(20)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert peak["max"] <= 3, f"semaphore breached: peak={peak['max']} > 3"


def test_C2_build_worker_pool_applies_per_model_caps(monkeypatch):
    """build_worker_pool wraps each model at its clean-concurrency cap."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    import _common
    # avoid real QwenWorker (needs DASHSCOPE key) — patch it
    from arch_policy.executor.multi_agent import Worker, WorkerOutput
    class _Fake(Worker):
        def __init__(self, model=None, **k): self.model = model
        def chat(self, s, u, max_new_tokens=512):
            return WorkerOutput(text="x", n_input_tokens=1, n_output_tokens=1)
    monkeypatch.setattr("arch_policy.QwenWorker", _Fake)
    pool = _common.build_worker_pool(
        ("qwen3.7-max", "glm-5.1", "deepseek-v4-pro", "unknown-x"),
        timeout=60, temperature=0.0, thinking=False)
    assert pool["qwen3.7-max"].max_concurrency == 128
    assert pool["glm-5.1"].max_concurrency == 50
    assert pool["deepseek-v4-pro"].max_concurrency == 32
    assert pool["unknown-x"].max_concurrency == 32  # conservative default


# ---------------------------------------------------------------------------
# H1 — LiveCodeBench functional grading
# ---------------------------------------------------------------------------

def test_H1_functional_correct_solution_passes():
    from arch_policy.reward.grade import grade_livecodebench
    import json
    code = "```python\nclass Solution:\n    def add(self, a, b):\n        return a + b\n```"
    tests = json.dumps([
        {"input": "2\n3", "output": "5", "testtype": "functional",
         "metadata": {"func_name": "add"}},
        {"input": "10\n-4", "output": "6", "testtype": "functional",
         "metadata": {"func_name": "add"}},
    ])
    assert grade_livecodebench(code, {"tests": tests}) == 1.0


def test_H1_functional_wrong_solution_fails():
    from arch_policy.reward.grade import grade_livecodebench
    import json
    code = "```python\nclass Solution:\n    def add(self, a, b):\n        return a - b\n```"
    tests = json.dumps([
        {"input": "2\n3", "output": "5", "testtype": "functional",
         "metadata": {"func_name": "add"}},
    ])
    assert grade_livecodebench(code, {"tests": tests}) == 0.0


def test_H1_functional_list_args_and_return():
    from arch_policy.reward.grade import grade_livecodebench
    import json
    code = ("```python\nclass Solution:\n"
            "    def merge(self, a, b):\n        return sorted(a + b)\n```")
    tests = json.dumps([
        {"input": "[3, 1]\n[2]", "output": "[1, 2, 3]", "testtype": "functional",
         "metadata": {"func_name": "merge"}},
    ])
    assert grade_livecodebench(code, {"tests": tests}) == 1.0


def test_H1_stdin_still_works_alongside_functional_routing():
    """The stdin path must be unaffected — a stdin row routes to the
    pipe harness, not the functional one."""
    from arch_policy.reward.grade import grade_livecodebench
    import json
    code = "```python\nprint(int(input()) * 2)\n```"
    tests = json.dumps([{"input": "21\n", "output": "42", "testtype": "stdin"}])
    assert grade_livecodebench(code, {"tests": tests}) == 1.0


# ---------------------------------------------------------------------------
# M1 — 429 / rate-limit retriable even when message says "quota"
# ---------------------------------------------------------------------------

def test_M1_rate_limit_with_quota_word_is_retriable(monkeypatch):
    """A 429 whose message contains 'quota' must NOT be classified
    non-retriable (the small-bucket vendors 429 under burst)."""
    from arch_policy.executor.qwen_worker import QwenWorker
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake")

    attempts = {"n": 0}
    from types import SimpleNamespace as NS

    def _ok_resp():
        msg = NS(content="ANSWER: ok", reasoning_content=None)
        return NS(choices=[NS(message=msg, finish_reason="stop")],
                  usage=NS(prompt_tokens=1, completion_tokens=1))

    def _create(**kw):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("Error 429: Requests rate limit exceeded, "
                               "free quota used up")
        return _ok_resp()

    client = NS(chat=NS(completions=NS(create=_create)))
    w = QwenWorker(api_key="fake", max_retries=4, retry_initial_delay=0.0)
    w._client = client
    out = w.chat("s", "u")
    assert attempts["n"] == 2, f"should have retried the 429; attempts={attempts['n']}"
    assert out.text == "ANSWER: ok"


def test_M1_real_quota_billing_still_non_retriable(monkeypatch):
    """A genuine billing/quota error (no rate-limit marker) stays
    non-retriable — we shouldn't hammer on a dead key."""
    from arch_policy.executor.qwen_worker import QwenWorker
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake")
    attempts = {"n": 0}
    from types import SimpleNamespace as NS

    def _create(**kw):
        attempts["n"] += 1
        raise RuntimeError("Insufficient quota / billing suspended")

    w = QwenWorker(api_key="fake", max_retries=4, retry_initial_delay=0.0)
    w._client = NS(chat=NS(completions=NS(create=_create)))
    out = w.chat("s", "u")
    assert attempts["n"] == 1, f"billing error must NOT retry; attempts={attempts['n']}"
    assert "error" in out.text.lower()


# ---------------------------------------------------------------------------
# R3b — M1's rate-limited-first guard ported to GpuGeek + DeepSeek workers.
# The judge runs on GpuGeekWorker; both lacked the guard qwen_worker has, so
# a 429 whose text also says "quota" was misrouted non-retriable → for the
# judge that becomes GRADE_ERROR → eng-invalid under burst.
# ---------------------------------------------------------------------------

def _ns_ok(text="ok"):
    from types import SimpleNamespace as NS
    msg = NS(content=text, reasoning_content=None)
    return NS(choices=[NS(message=msg, finish_reason="stop")],
              usage=NS(prompt_tokens=1, completion_tokens=1))


def test_R3b_gpugeek_429_with_quota_is_retriable():
    """GpuGeek (the judge vendor): a 429 mentioning 'quota' must retry."""
    from types import SimpleNamespace as NS
    from arch_policy.executor.gpugeek_worker import GpuGeekWorker
    attempts = {"n": 0}

    def _create(**kw):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("Error 429: rate limit exceeded, free quota used up")
        return _ns_ok()

    w = GpuGeekWorker(api_key="fake", max_retries=4, retry_initial_delay=0.0)
    w._client = lambda key: NS(chat=NS(completions=NS(create=_create)))
    out = w.chat("s", "u")
    assert attempts["n"] == 2, f"429 must retry; attempts={attempts['n']}"
    assert out.text == "ok"


def test_R3b_gpugeek_pure_billing_still_non_retriable():
    from types import SimpleNamespace as NS
    from arch_policy.executor.gpugeek_worker import GpuGeekWorker
    attempts = {"n": 0}

    def _create(**kw):
        attempts["n"] += 1
        raise RuntimeError("Insufficient quota / billing suspended")

    w = GpuGeekWorker(api_key="fake", max_retries=4, retry_initial_delay=0.0)
    w._client = lambda key: NS(chat=NS(completions=NS(create=_create)))
    out = w.chat("s", "u")
    assert attempts["n"] == 1, f"billing must NOT retry; attempts={attempts['n']}"


def test_R3b_deepseek_429_with_quota_is_retriable():
    from types import SimpleNamespace as NS
    from arch_policy.executor.deepseek_worker import DeepSeekWorker
    attempts = {"n": 0}

    def _create(**kw):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("429 Too Many Requests (quota)")
        return _ns_ok()

    w = DeepSeekWorker(api_key="fake", max_retries=4, retry_initial_delay=0.0)
    w._client = NS(chat=NS(completions=NS(create=_create)))
    out = w.chat("s", "u")
    assert attempts["n"] == 2, f"429 must retry; attempts={attempts['n']}"
    assert out.text == "ok"


def test_R3b_deepseek_pure_billing_still_non_retriable():
    from types import SimpleNamespace as NS
    from arch_policy.executor.deepseek_worker import DeepSeekWorker
    attempts = {"n": 0}

    def _create(**kw):
        attempts["n"] += 1
        raise RuntimeError("Insufficient quota / billing suspended")

    w = DeepSeekWorker(api_key="fake", max_retries=4, retry_initial_delay=0.0)
    w._client = NS(chat=NS(completions=NS(create=_create)))
    out = w.chat("s", "u")
    assert attempts["n"] == 1, f"billing must NOT retry; attempts={attempts['n']}"
