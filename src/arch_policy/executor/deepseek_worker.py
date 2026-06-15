"""DeepSeek native API worker (https://api.deepseek.com).

Single-vendor counterpart to GpuGeekWorker. Use when paying out of pocket:
deepseek-v4-flash with thinking disabled is the cheapest reliable path
(~$0.00014 in / $0.00028 out per 1K tokens; ~0.7s round-trip on a small
prompt).

Auth: read DEEPSEEK_API_KEY from env (or pass api_key=).

Thinking control (per DeepSeek docs):
  default ON — model emits a `reasoning_content` block of 500-1500 tokens
               before `content`, billed at the same per-token rate.
  thinking=False (our default) — pass `extra_body={"thinking":{"type":"disabled"}}`
               so the request goes through the cheap non-CoT path.

The OpenAI SDK forwards `extra_body` into the top-level JSON body, which
matches what DeepSeek's HTTP API expects (verified May 2026).
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from .multi_agent import Worker, WorkerOutput


@dataclass
class DeepSeekWorker(Worker):
    """Worker proxying chat to DeepSeek's native /chat/completions endpoint.

    Thread-safe: the OpenAI client + underlying httpx pool are shared
    across threads per library guarantees; we size the pool to 512
    connections so high-concurrency GRPO steps don't stall.
    """

    model: str = "deepseek-v4-flash"
    api_key: Optional[str] = None
    base_url: str = "https://api.deepseek.com"
    timeout: float = 120.0
    temperature: float = 0.0
    max_retries: int = 6
    retry_initial_delay: float = 1.0
    # OFF by default: cuts both latency and output tokens. Flip to True
    # via --worker_thinking for benchmark eval where reasoning matters.
    thinking: bool = False

    _client: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "DeepSeekWorker: no api_key passed and DEEPSEEK_API_KEY env "
                "var not set. Get a key from https://platform.deepseek.com."
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
            # Disable openai-internal retries — we have our own loop.
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
                kw = dict(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    max_tokens=max_new_tokens,
                    temperature=self.temperature,
                )
                if self.thinking:
                    # Per DeepSeek docs: enabled + high effort. (low/medium
                    # are silently mapped to high; xhigh maps to max.)
                    kw["extra_body"] = {"thinking": {"type": "enabled"}}
                    kw["reasoning_effort"] = "high"
                else:
                    kw["extra_body"] = {"thinking": {"type": "disabled"}}
                resp = self._ensure_client().chat.completions.create(**kw)
                choice = resp.choices[0]
                text = (choice.message.content or "").strip()
                # Extract reasoning_content into a SEPARATE field. Per the
                # WorkerOutput reasoning-isolation contract, reasoning MUST
                # NEVER be propagated to other agents / Synth / scratchpad.
                # We expose it here only for opt-in telemetry; the agent
                # layer never reads it. If content is empty (budget
                # exhausted by reasoning) we deliberately let `text` stay
                # empty — agent.py treats that as an empty_text protocol
                # failure, which is the correct outcome (no contribution
                # from this turn, rather than leaking CoT into the trace).
                rc = getattr(choice.message, "reasoning_content", None) or ""
                reasoning = rc.strip() or None
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
                # non-retriable branch below (which keys on "quota"). DeepSeek-
                # V4-Pro has the smallest concurrency bucket (32) → most prone
                # to 429 under burst.
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
                text=f"[DeepSeekWorker error: {type(last_error).__name__}: {last_error}]",
                n_input_tokens=0, n_output_tokens=0,
            )
        return WorkerOutput(text="", n_input_tokens=0, n_output_tokens=0)


__all__ = ["DeepSeekWorker"]
