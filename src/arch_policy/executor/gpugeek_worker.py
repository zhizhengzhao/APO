"""GpuGeek-backed Worker — multi-vendor LLM gateway via a single API key.

GpuGeek (https://api.gpugeek.com) proxies multiple frontier models behind
the OpenAI Chat Completions protocol. This worker covers any model whose
GpuGeek model id is accepted at `/v1/chat/completions`.

Models / native protocols intentionally NOT supported here:
  - Claude (Anthropic /v1/messages): use a dedicated Anthropic worker
    if needed. GpuGeek's Claude proxy required raw httpx and a separate
    payload shape, which doubled this file's size for one model.
  - Gemini (Google /v1beta:generateContent): same story.
Both were dropped during the May 2026 minimalism pass.

Auth: read GPUGEEK_API_KEY from env (or pass `api_key=...`). Supports
multiple keys for higher rate-limit budget — pass `api_keys=[...]` or
comma-separated `api_key="k1,k2,..."` or env `GPUGEEK_API_KEY="k1,k2,..."`.
Calls dispatch round-robin across keys, one cached client per key.

Thread safety: instances are safe to share across threads (GRPO uses a
ThreadPoolExecutor). httpx connection pools are sized for B*G concurrent
arch runs × multiple LLM calls each.
"""

from __future__ import annotations

import itertools
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from .multi_agent import Worker, WorkerOutput


_WARNED_THINKING_UNKNOWN: set[str] = set()


def _apply_thinking_control(model: str, thinking: bool, kw: dict) -> None:
    """Per-model reasoning knob for the OpenAI-compat chat API.

      DeepSeek-V4-Flash / GPT-5.5  →  reasoning_effort: "none" | "high"
      DeepSeek-V4-Pro              →  extra_body.thinking.type: "disabled"
                                      (rejects reasoning_effort="none")
      GPT-5.1 / Claude-4.7-opus / Gemini-3.1-pro
                                   →  reasoning always on, no off-switch.
                                      Caller MUST size max_tokens for
                                      hidden reasoning + visible content
                                      (≥ 64 verified; we use 512 in prod).
      unknown                      →  pass-through + one-time WARN
    """
    if model == "Vendor3/DeepSeek-V4-Flash":
        kw["reasoning_effort"] = "high" if thinking else "none"
    elif model == "Vendor3/DeepSeek-V4-Pro":
        if thinking:
            kw["reasoning_effort"] = "high"
        else:
            kw.setdefault("extra_body", {})["thinking"] = {"type": "disabled"}
    elif model == "Vendor2/GPT-5.5":
        kw["reasoning_effort"] = "high" if thinking else "none"
    elif model in (
        "Vendor2/GPT-5.1",
        "Vendor2/Claude-4.7-opus",
        "Vendor2/Gemini-3.1-pro",
    ):
        pass   # always-on reasoning, no knob
    else:
        if model not in _WARNED_THINKING_UNKNOWN:
            _WARNED_THINKING_UNKNOWN.add(model)
            print(f"[GpuGeekWorker] WARN model={model!r} not in "
                  f"thinking-control dispatch; passing through with no "
                  f"reasoning knob. Ensure max_tokens leaves room for "
                  f"visible content if this model has hidden reasoning.",
                  flush=True)


@dataclass
class GpuGeekWorker(Worker):
    """Worker proxying chat to GpuGeek's OpenAI-compat endpoint.

    Usage:
        w = GpuGeekWorker(model="Vendor3/DeepSeek-V4-Flash")
        executor = MultiAgentExecutor(worker=w)
    """

    model: str = "Vendor3/DeepSeek-V4-Flash"
    api_key: Optional[str] = None              # single key OR comma-list
    api_keys: Optional[list[str]] = None       # explicit list (preferred)
    base_url: str = "https://api.gpugeek.com"
    timeout: float = 45.0
    temperature: float = 0.0
    max_retries: int = 6
    retry_initial_delay: float = 1.0
    # OFF by default for speed; flip via `--worker_thinking` for benchmark
    # eval where reasoning quality matters.
    thinking: bool = False

    _key_pool: list[str] = field(default_factory=list, init=False, repr=False)
    _openai_clients: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _rr_cycle: Optional[object] = field(default=None, init=False, repr=False)
    _rr_lock: object = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        # Key pool resolution: api_keys > api_key (comma-split) > env.
        if self.api_keys:
            keys = list(self.api_keys)
        elif self.api_key:
            keys = [k.strip() for k in self.api_key.split(",") if k.strip()]
        else:
            env = os.environ.get("GPUGEEK_API_KEY", "")
            keys = [k.strip() for k in env.split(",") if k.strip()]
        if not keys:
            raise RuntimeError(
                "GpuGeekWorker: no api_key(s) passed and GPUGEEK_API_KEY env "
                "var not set. Get a key from https://gpugeek.com."
            )
        seen, pool = set(), []
        for k in keys:
            if k not in seen:
                seen.add(k); pool.append(k)
        self._key_pool = pool
        self.api_key = pool[0]   # keep singular field meaningful
        self._rr_cycle = itertools.cycle(pool)

    def _next_key(self) -> str:
        """Thread-safe round-robin key selection.

        GpuGeek imposes no per-key concurrency limit (verified up to 256
        concurrent requests); throughput is bounded by upstream model
        latency, not gateway rate-limiting.
        """
        with self._rr_lock:
            return next(self._rr_cycle)

    @property
    def n_keys(self) -> int:
        return len(self._key_pool)

    def _client(self, key: str):
        client = self._openai_clients.get(key)
        if client is not None:
            return client
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai package required: `pip install openai>=1.30`") from e
        try:
            import httpx
            http_client = httpx.Client(
                limits=httpx.Limits(
                    max_connections=512,
                    max_keepalive_connections=128,
                    keepalive_expiry=30.0,
                ),
                timeout=httpx.Timeout(self.timeout, connect=15.0),
            )
            client = OpenAI(
                api_key=key, base_url=f"{self.base_url}/v1",
                http_client=http_client, max_retries=0,
            )
        except Exception:
            client = OpenAI(
                api_key=key, base_url=f"{self.base_url}/v1",
                timeout=self.timeout, max_retries=0,
            )
        self._openai_clients[key] = client
        return client

    def chat(self, system: str, user: str, max_new_tokens: int = 512) -> WorkerOutput:
        delay = self.retry_initial_delay
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                client = self._client(self._next_key())
                kw = dict(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=max_new_tokens,
                    temperature=self.temperature,
                )
                _apply_thinking_control(self.model, self.thinking, kw)
                resp = client.chat.completions.create(**kw)
                choice = resp.choices[0]
                text = (choice.message.content or "").strip()
                # Per the WorkerOutput reasoning-isolation contract, never
                # propagate reasoning_content into `text`. Empty content
                # (whole budget burned by CoT) is treated as an empty
                # reply by the agent layer.
                rc_raw = getattr(choice.message, "reasoning_content", None) or ""
                reasoning = rc_raw.strip() or None
                usage = resp.usage
                in_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                out_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                # finish_reason='length' → see WorkerOutput.truncated.
                truncated = (str(getattr(choice, "finish_reason", "") or "").lower()
                             == "length")
                return WorkerOutput(
                    text=text, n_input_tokens=in_tokens,
                    n_output_tokens=out_tokens, reasoning=reasoning,
                    truncated=truncated,
                )
            except Exception as e:  # noqa: BLE001 — broad retry
                last_error = e
                msg = str(e).lower()
                # Rate limits (429) are ALWAYS retriable + checked first, so a
                # "rate limit … quota" message can't be misrouted to the
                # non-retriable branch below (which keys on "quota"). Critical
                # for the judge: 128 traces finishing together burst the judge
                # endpoint, and a misclassified 429 → GRADE_ERROR → eng-invalid.
                rate_limited = (
                    "429" in msg
                    or "rate limit" in msg
                    or "too many requests" in msg
                    or "ratelimit" in msg
                    or type(e).__name__ == "RateLimitError"
                )
                non_retriable = (not rate_limited) and (
                    "unauthorized" in msg
                    or "invalid api key" in msg
                    or "quota" in msg
                    or "billing" in msg
                    or "bad request" in msg
                )
                if non_retriable or attempt == self.max_retries - 1:
                    break
                # Per-thread jitter so concurrent retries DO NOT line up
                # (see qwen_worker.py for full thundering-herd rationale).
                jitter = 0.5 + (hash((threading.get_ident(),
                                       time.time_ns(), attempt)) % 1000) / 1000.0
                time.sleep(delay * jitter)
                delay *= 2

        if last_error is not None:
            return WorkerOutput(
                text=f"[GpuGeekWorker error: {type(last_error).__name__}: {last_error}]",
                n_input_tokens=0, n_output_tokens=0,
            )
        return WorkerOutput(text="", n_input_tokens=0, n_output_tokens=0)


__all__ = ["GpuGeekWorker"]
