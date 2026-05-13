"""OpenAI / OpenAI-compatible API worker for the executor.

This is the worker we plan to use for actual training and evaluation: it
calls a remote API (OpenAI, DeepSeek, vLLM-compatible) instead of running
inference locally. Reasons we'd want this:

  - Lets us run baselines under the *same* worker as G-Designer/MaAS/etc.
  - Decouples Architecture-Policy training from local GPU availability.
  - Cheaper than running a 70B model locally for ablation experiments.

Auth:
  Set env vars before launching:
    OPENAI_API_KEY   = "sk-..."
    OPENAI_BASE_URL  = "https://api.openai.com/v1"  (or proxy)
  Or pass explicitly to the constructor.

Throttling:
  We use simple in-process retry-with-exponential-backoff. For high-throughput
  parallel runs use a queue + worker pool externally.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

from .multi_agent import Worker, WorkerOutput


@dataclass
class OpenAIWorker(Worker):
    """A `Worker` that proxies chat to an OpenAI-compatible chat-completions endpoint.

    Usage::

        worker = OpenAIWorker(model="gpt-4o-mini", api_key="sk-...")
        executor = MultiAgentExecutor(worker=worker)

    The class is intentionally simple — single thread, blocking, exponential
    backoff on transient errors.
    """

    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 120.0
    temperature: float = 0.0
    max_retries: int = 4
    retry_initial_delay: float = 2.0

    # internal: lazily-initialized OpenAI client
    _client: Optional[object] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if self.api_key is None:
                raise RuntimeError(
                    "OpenAIWorker: no api_key passed and OPENAI_API_KEY env var not set."
                )
        if self.base_url is None:
            self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # ------------------------------------------------------------------
    def _client_singleton(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "openai package not installed. `pip install openai>=1.30`"
                ) from e
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        return self._client

    # ------------------------------------------------------------------
    def chat(self, system: str, user: str, max_new_tokens: int = 512) -> WorkerOutput:
        """Single chat call. Retries on transient errors."""
        client = self._client_singleton()
        delay = self.retry_initial_delay
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=max_new_tokens,
                    temperature=self.temperature,
                )
                choice = resp.choices[0]
                text = (choice.message.content or "").strip()
                usage = resp.usage
                in_tokens = int(getattr(usage, "prompt_tokens", 0))
                out_tokens = int(getattr(usage, "completion_tokens", 0))
                return WorkerOutput(text=text, n_input_tokens=in_tokens, n_output_tokens=out_tokens)
            except Exception as e:  # noqa: BLE001 — we want broad retry
                last_error = e
                msg = str(e).lower()
                # Only retry on retriable errors (rate limits, 5xx, timeouts).
                retriable = (
                    "rate limit" in msg
                    or "timeout" in msg
                    or "503" in msg
                    or "502" in msg
                    or "429" in msg
                )
                if not retriable or attempt == self.max_retries - 1:
                    break
                time.sleep(delay)
                delay *= 2

        # Final failure → return empty output rather than crash.
        # Caller can detect via empty text + 0 tokens.
        if last_error is not None:
            return WorkerOutput(
                text=f"[OpenAIWorker error: {type(last_error).__name__}: {last_error}]",
                n_input_tokens=0,
                n_output_tokens=0,
            )
        return WorkerOutput(text="", n_input_tokens=0, n_output_tokens=0)


__all__ = ["OpenAIWorker"]
