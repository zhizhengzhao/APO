"""Aliyun Bailian (DashScope) worker for Qwen models.

Single-vendor worker for the Qwen family via the OpenAI-compatible endpoint
at `https://dashscope.aliyuncs.com/compatible-mode/v1`. Verified May 2026
against qwen3.7-max / qwen-flash with thinking=False.

Auth: read DASHSCOPE_API_KEY from env (or pass api_key=).

Thinking control (Hybrid Qwen3 family — qwen3.5-plus / qwen3.6-plus / qwen3.7-max):
  default ON  — model emits a `reasoning_content` block before `content`,
                billed at the same per-token rate.
  thinking=False (our default) — `extra_body={"enable_thinking": False}`
                routes the request through the non-CoT path. ~3-5x cheaper
                and faster on short prompts.

Cost note (qwen3.7-max, 5-off promo, 2026 May):
  - input  ¥6/M tokens  ≈ $0.00086 / 1K
  - output ¥18/M tokens ≈ $0.00257 / 1K
  ~7-8x more expensive than DeepSeek-V4-Flash, but throughput at high
  concurrency is much smoother (p99 ~1.13s @ 64-way, vs GpuGeek which
  degrades). See `provider-docs/qwen/pricing-and-limits.md` for the full
  table + empirical concurrency probe.

Cheaper qwen alternatives (same Worker class, just pass `model=`):
  qwen-flash      ¥0.2 / ¥1.5  per M tokens  (~30x cheaper than 3.7-max)
  qwen3.5-plus    ¥0.8 / ¥4.8  per M tokens  (~7x cheaper)
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from .multi_agent import Worker, WorkerOutput


@dataclass
class QwenWorker(Worker):
    """Worker proxying chat to DashScope's OpenAI-compat /chat/completions.

    Thread-safe: the OpenAI client + underlying httpx pool are shared
    across threads per library guarantees; we size the pool to 512
    connections so high-concurrency GRPO steps don't stall.
    """

    model: str = "qwen3.7-max"
    api_key: Optional[str] = None
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    timeout: float = 120.0
    temperature: float = 0.0
    max_retries: int = 6
    retry_initial_delay: float = 1.0
    # OFF by default: routes through the non-thinking branch on Hybrid
    # Qwen3 models (qwen3.5-plus / qwen3.6-plus / qwen3.7-max). For
    # thinking-only variants (QwQ / `-thinking` suffixes) this flag is
    # effectively ignored by the backend.
    thinking: bool = False
    # Thinking-mode-only: cap CoT tokens. None = unlimited (per docs,
    # qwen3.7-max can spend up to 256K CoT tokens). Set a small value to
    # bound cost when you do enable thinking. Ignored when thinking=False.
    thinking_budget: Optional[int] = None

    _client: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "QwenWorker: no api_key passed and DASHSCOPE_API_KEY env "
                "var not set. Get one from https://bailian.console.aliyun.com."
            )

    def _ensure_client(self):
        if self._client is not None:
            return self._client
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
            self._client = OpenAI(
                api_key=self.api_key, base_url=self.base_url,
                http_client=http_client,
                max_retries=0,
            )
        except Exception:
            self._client = OpenAI(
                api_key=self.api_key, base_url=self.base_url,
                timeout=self.timeout, max_retries=0,
            )
        return self._client

    def chat(self, system: str, user: str, max_new_tokens: int = 512) -> WorkerOutput:
        delay = self.retry_initial_delay
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                extra: dict = {"enable_thinking": bool(self.thinking)}
                if self.thinking and self.thinking_budget is not None:
                    extra["thinking_budget"] = int(self.thinking_budget)
                kw = dict(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    max_tokens=max_new_tokens,
                    temperature=self.temperature,
                    extra_body=extra,
                )
                resp = self._ensure_client().chat.completions.create(**kw)
                choice = resp.choices[0]
                text = (choice.message.content or "").strip()
                # Per the WorkerOutput reasoning-isolation contract, never
                # propagate reasoning_content into `text`. Empty content
                # (whole budget burned by CoT) is treated as an empty
                # reply by the agent layer — that's the correct outcome
                # rather than leaking CoT into the trace.
                rc = getattr(choice.message, "reasoning_content", None) or ""
                reasoning = rc.strip() or None
                usage = resp.usage
                in_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                out_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                # finish_reason='length' → mid-token cut at max_tokens.
                # agent.py treats this as worker_error so the trace is
                # eng_valid-masked rather than committing a half-sentence
                # reply as `submit_implicit`.
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
                # Rate limits (429) are ALWAYS retriable + checked first,
                # so a "rate limit … quota" message can't be misrouted to
                # the non-retriable branch below (which keys on "quota").
                # Critical for the multi-model pool where small-bucket
                # vendors (GLM 50 / DeepSeek 32) 429 under burst.
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
                    or "ip access denied" in msg
                )
                if non_retriable or attempt == self.max_retries - 1:
                    break
                # Per-thread jitter so concurrent traces DON'T retry in
                # lockstep. The previous formula `hash((id(self), attempt))`
                # used the SAME shared instance id across all threads + the
                # same attempt count, so every concurrent retry computed
                # the same jitter and woke up simultaneously — a
                # thundering-herd amplifier under API 5xx surge. Now we
                # mix in the current thread id + a nanosecond clock so
                # 32 concurrent retries get 32 distinct sleep durations.
                jitter = 0.5 + (hash((threading.get_ident(),
                                       time.time_ns(), attempt)) % 1000) / 1000.0
                time.sleep(delay * jitter)
                delay *= 2

        if last_error is not None:
            return WorkerOutput(
                text=f"[QwenWorker error: {type(last_error).__name__}: {last_error}]",
                n_input_tokens=0, n_output_tokens=0,
            )
        return WorkerOutput(text="", n_input_tokens=0, n_output_tokens=0)


__all__ = ["QwenWorker"]
