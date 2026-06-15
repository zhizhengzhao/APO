"""Multi-agent executor — main loop.

Naming: episode > cycle > turn > step.
  episode — one MultiAgentExecutor.run call.
  cycle   — one pass through the PL-sampled sequence.
  turn    — one agent's slot in a cycle (Agent.run invocation).
  step    — one LLM call inside a turn's ReAct loop.

Main loop:

  for cycle in 1 .. safety_max_cycles:
      for slot in arch.sequence:
          incoming = chronological slice of trace.messages, filtered by
                     edges (+ self auto-visible, + history_cycles cap,
                     + skipped excluded)
          turn_out = agent[slot].run(task, incoming, cycle, turn_idx)
          record turn (skipped turns get no content but still get an
          AgentMessage row for telemetry)
      verdict = synth.judge(task, full trace.messages)
      if verdict.is_done: final_answer = verdict.answer; break
  else:
      final_answer = heuristic_extract(trace.messages)   # safety cap fallback

Worker abstractions live here (`Worker`, `MockWorker`). The OpenAI- and
GpuGeek-compatible workers live in sibling modules.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from ..architecture.sampler import ConcreteArch
from ..config import ARCH, ArchSpec
from .trace import AgentMessage, ExecutionTrace


# ---------------------------------------------------------------------------
# Worker abstraction
# ---------------------------------------------------------------------------

@dataclass
class WorkerOutput:
    """Result of one worker.chat call.

    `text`      — public reply, parsed by the agent for ACTION blocks.
    `reasoning` — internal CoT (DeepSeek `reasoning_content`, Claude
                  thinking blocks); None when the provider hides it.
                  CONTRACT: never propagate to other agents / Synth /
                  next-step scratchpad. Enforced by
                  `tests/test_reasoning_isolation.py`.
    `truncated` — True iff finish_reason=='length' (mid-token cut at
                  max_tokens cap). The agent layer treats this as
                  `worker_error` so eng_valid masks the sample from the
                  GRPO gradient — max_new_tokens is a meta-confound the
                  architecture cannot control.
    """
    text: str
    n_input_tokens: int = 0
    n_output_tokens: int = 0
    reasoning: str | None = None
    truncated: bool = False    # finish_reason == 'length' (mid-token cut)


class Worker(ABC):
    """Anything that turns (system, user) into (text, token counts)."""

    @abstractmethod
    def chat(self, system: str, user: str, max_new_tokens: int = 512) -> WorkerOutput: ...


class ConcurrencyLimitedWorker(Worker):
    """Wraps a Worker with a bounded semaphore so total in-flight calls to
    one model never exceed its provider's clean-concurrency limit.

    Multi-model runs route many concurrent traces' agents to whichever
    model the head prefers; without a per-model cap a burst onto a
    small-bucket vendor (GLM 50 / DeepSeek 32) overruns it → 429s → the
    head learns to avoid that vendor for being throttled, not for being
    worse. The semaphore makes excess calls WAIT rather than fail, so the
    model dimension's reward reflects quality, not rate-limit luck.
    """

    def __init__(self, inner: Worker, max_concurrency: int):
        import threading as _t
        self.inner = inner
        self.max_concurrency = max_concurrency
        self._sem = _t.BoundedSemaphore(max_concurrency)

    def chat(self, system: str, user: str, max_new_tokens: int = 512) -> WorkerOutput:
        with self._sem:
            return self.inner.chat(system, user, max_new_tokens=max_new_tokens)


# ---------------------------------------------------------------------------
# Mock worker (deterministic, no LLM) — for tests
# ---------------------------------------------------------------------------

class MockWorker(Worker):
    """Deterministic mock that mirrors role-suffix conventions.

    By default it returns a Solver-style 'Candidate: <fake_answer>' suffix.
    Pass `force_synth_done=True` to make it always reply 'ANSWER: <fake>' so
    the executor terminates after one cycle (great for fast smoke tests).
    """

    def __init__(
        self,
        fake_answer: str = "42",
        force_synth_done: bool = True,
    ) -> None:
        self.fake_answer = fake_answer
        self.force_synth_done = force_synth_done
        self.calls: list[tuple[str, str]] = []

    def chat(self, system: str, user: str, max_new_tokens: int = 512) -> WorkerOutput:
        self.calls.append((system, user))
        if self.force_synth_done and "You are a Synth" in system:
            text = f"ANSWER: {self.fake_answer}"
        elif "You are a Synth" in system:
            text = "CONTINUE"
        else:
            # Plain-text reply: parse_action returns None → implicit submit.
            # Mirrors how real LLMs typically respond.
            if "Verifier" in system:
                text = f"Verified: {self.fake_answer}"
            elif "Refiner" in system:
                text = f"Refined: {self.fake_answer}"
            elif "Tester" in system:
                text = f"TESTS RESULT: pass (mock); answer {self.fake_answer}"
            elif "Researcher" in system:
                text = f"Findings: known facts say {self.fake_answer}"
            elif "Planner" in system:
                text = "PLAN:\n  1. compute the answer\n  2. verify"
            elif "Expert" in system:
                text = f"Candidate: {self.fake_answer}"
            elif "Critic" in system:
                text = "Critique: looks fine"
            else:
                text = f"Candidate: {self.fake_answer}"
        return WorkerOutput(
            text=text,
            n_input_tokens=len((system + user).split()),
            n_output_tokens=len(text.split()),
        )


# ---------------------------------------------------------------------------
# Executor (AgentMessage + ExecutionTrace live in .trace)
# ---------------------------------------------------------------------------

class MultiAgentExecutor:
    """Run a `ConcreteArch` against a task.

    Single-model by default (one `worker` for every agent slot + Synth).
    For multi-model runs, pass `worker_pool={model_name: Worker}`: an agent
    at slot s then runs on `worker_pool[spec.model_names[arch.model[s]]]`,
    i.e. the model the head's 5th typed dimension chose for that slot.
    Synth always runs on the fixed `synth_worker`/`worker`.

    Args:
      worker: default LLM worker; used for every agent slot in single-model
        mode AND always for Synth.
      worker_pool: optional {model_name: Worker} for per-slot dispatch.
      synth_worker: optional override; if None, Synth uses `worker`.
      max_new_tokens_per_call: cap per worker.chat output; defaults to
        ArchSpec.safety_max_tokens_per_call.
      wall_clock_timeout_s: hard per-trace wall-clock cap (seconds; 0
        disables) — bounds total run time even under API tail-latency.
      max_llm_calls_per_trace: soft cap on total worker.chat calls (incl.
        Synth) inside one trace. 0 disables. When hit, the executor breaks
        the cycle loop and falls back to heuristic_extract; the trace is
        flagged hit_call_cap and counted as an arch-attributable cap
        (n_arch_caps_hit+=1). Reward signal is NOT masked — correctness of
        the partial answer drives advantage (bad-arch attribution).
      history_cycles: cap how many past cycles each agent's incoming
        prompt slice contains (0 = unlimited). Useful on long episodes
        with expensive models. Synth always sees the full trace.
      parallel_within_cycle: when True (default), decompose each cycle's
        active sequence into topological waves based on `arch.edges` and
        run all agents in a wave concurrently. Agents in the same wave
        have no edge between them so the parallel ordering is semantically
        identical to sequential. When False, falls back to strict
        single-threaded execution (debugging / reproducibility).
    """

    def __init__(
        self,
        worker: "Worker",
        spec: ArchSpec | None = None,
        max_new_tokens_per_call: int | None = None,
        synth_max_new_tokens: int = 1024,
        synth_worker: "Worker | None" = None,
        wall_clock_timeout_s: float = 300.0,
        max_llm_calls_per_trace: int = 0,
        history_cycles: int = 0,
        parallel_within_cycle: bool = True,
        worker_pool: "dict[str, Worker] | None" = None,
    ) -> None:
        if worker is None:
            raise ValueError("MultiAgentExecutor needs a `worker`")
        self.spec = spec or ARCH
        self.worker = worker
        # Per-slot model dispatch: when set, an agent at slot s runs on
        # worker_pool[spec.model_names[arch.model[s]]]. None ⇒ single-model
        # (every slot + Synth on `worker`). Keys must cover spec.model_names.
        self.worker_pool = worker_pool
        # Silent-degradation guard: a multi-model spec (n_models>1) with NO
        # worker_pool would dispatch every slot to the single `worker`, so the
        # sampled per-agent model never affects reward — the model dim then
        # trains on pure noise (no crash, just meaningless "model preference"
        # results). Fail loudly instead.
        if self.spec.n_models > 1 and worker_pool is None:
            raise ValueError(
                f"spec.n_models={self.spec.n_models} > 1 but worker_pool is None "
                f"— the per-agent model dimension would silently degrade to "
                f"single-worker (model choice wouldn't affect reward). Pass a "
                f"worker_pool covering {list(self.spec.model_names)}."
            )
        if worker_pool is not None:
            missing = [m for m in self.spec.model_names if m not in worker_pool]
            if missing:
                raise ValueError(
                    f"worker_pool missing workers for models {missing}; "
                    f"have {sorted(worker_pool)}"
                )
        self.synth_worker = synth_worker if synth_worker is not None else worker
        self.max_new_tokens = (
            max_new_tokens_per_call
            if max_new_tokens_per_call is not None
            else self.spec.safety_max_tokens_per_call
        )
        self.synth_max_new_tokens = synth_max_new_tokens
        self.wall_clock_timeout_s = wall_clock_timeout_s
        if max_llm_calls_per_trace < 0:
            raise ValueError(
                f"max_llm_calls_per_trace must be >= 0, got {max_llm_calls_per_trace}"
            )
        self.max_llm_calls_per_trace = int(max_llm_calls_per_trace)
        if history_cycles < 0:
            raise ValueError(f"history_cycles must be >= 0, got {history_cycles}")
        self.history_cycles = int(history_cycles)
        self.parallel_within_cycle = bool(parallel_within_cycle)

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_waves(arch: "ConcreteArch",
                       active_idx: list[int]) -> list[list[int]]:
        """Decompose the cycle's speaking sequence into topological waves.

        Slot Y is a within-cycle predecessor of slot X iff:
          - Y appears BEFORE X in `active_idx` (sequence order), AND
          - arch.edges[Y, X] is True (Y feeds into X).

        Two slots in the same wave have no edge between them (within this
        cycle), so running them concurrently is semantically identical to
        running them sequentially in any order.

        Returns: list of waves. Each wave is a list of slot indices, ordered
        as they appear in `active_idx` for determinism. Concatenating waves
        in order yields a permutation of `active_idx`.

        Guaranteed: at least one slot enters every iteration (predecessors
        are strictly ordered by sequence, so the dep graph is a DAG → no
        deadlock possible).
        """
        # Edge lookup is O(1) per (i, j). Build a quick "preds of slot in
        # active_idx" map once.
        pos = {s: i for i, s in enumerate(active_idx)}
        preds: dict[int, list[int]] = {}
        for slot in active_idx:
            preds[slot] = [
                y for y in active_idx
                if y != slot and pos[y] < pos[slot]
                and bool(arch.edges[y, slot].item())
            ]

        waves: list[list[int]] = []
        remaining = list(active_idx)
        while remaining:
            ready = [s for s in remaining
                     if all(p not in remaining for p in preds[s])]
            if not ready:
                # Unreachable in correct code: strict-sequence preds
                # guarantee `remaining[0]` has no in-remaining preds.
                # Fail LOUD (not silent fallback to remaining[0]) — a
                # silent recovery would run an agent before its preds
                # committed, breaking the DAG visibility contract.
                raise RuntimeError(
                    f"_compute_waves deadlock: remaining={remaining}, "
                    f"preds={preds}. Sequence-ordered predecessors must "
                    f"always yield at least one ready slot; if you see "
                    f"this, the wave-computation invariant is broken."
                )
            waves.append(ready)
            for s in ready:
                remaining.remove(s)
        return waves

    # ------------------------------------------------------------------
    @staticmethod
    def _build_incoming(
        msgs_snapshot: list["AgentMessage"],
        speaker_slot: int,
        arch: "ConcreteArch",
        cycle: int,
        history_cycles: int,
    ) -> list[tuple[int, str, int, str]]:
        """Filter `msgs_snapshot` to what `speaker_slot` should see.

        Rules (preserved verbatim from the original sequential code):
          - skipped messages dropped
          - self always visible (no self-edge required)
          - other slots require arch.edges[src, speaker] = True
          - optional cycle truncation by history_cycles (0 = unlimited)
        """
        incoming: list[tuple[int, str, int, str]] = []
        min_cycle = (
            cycle - history_cycles + 1
            if history_cycles > 0 else -1
        )
        for m in msgs_snapshot:
            if m.skipped:
                continue
            if history_cycles > 0 and m.cycle < min_cycle:
                continue
            if m.slot == speaker_slot:
                pass
            elif not arch.edges[m.slot, speaker_slot].item():
                continue
            incoming.append((m.slot, m.role, m.cycle, m.text))
        return incoming

    @staticmethod
    def _run_one_turn(
        agent: "Agent",
        task: str,
        incoming: list[tuple[int, str, int, str]],
        cycle: int,
        turn_idx: int,
        n_turns: int,
        wall_deadline: float | None,
        speaker_slot: int,
    ) -> tuple["AgentMessage", "AgentTurnOutput"]:
        """Pure-compute per-turn: call agent.run, build AgentMessage.

        Runs in an inner ThreadPoolExecutor worker thread for multi-slot
        waves. To preserve telemetry across the thread boundary:

          - tools' stub counters live in `threading.local()`, so each
            inner worker thread starts with a fresh count. We reset at
            turn start + snapshot at turn end, then attach the snapshot
            to `turn_out` for `_commit_turn` to merge under the trace
            lock. Without this, multi-wave traces lose all stub
            attribution (the main thread's TLS never gets touched).
          - agent.run exceptions are also packaged into `turn_out` as a
            structured `run_error` dict instead of disappearing into a
            stderr printout, so `details.jsonl` keeps the full
            traceback for forensics.
        """
        from .tools import (
            reset_search_stub_counts, snapshot_search_stub_counts,
        )
        # Reset stub TLS counter BEFORE the turn so any leakage from
        # a prior turn on the same worker thread can't mis-attribute
        # failures to this turn's architecture.
        reset_search_stub_counts()
        run_error_payload = None
        try:
            turn_out = agent.run(
                task=task,
                incoming=incoming,
                cycle=cycle,
                turn=turn_idx,
                n_turns=n_turns,
                wall_deadline=wall_deadline,
            )
        except Exception as e:  # noqa: BLE001
            # Tool bug, prompt-format crash, anything unexpected →
            # synthesize a worker_error turn. The trace continues, the
            # architecture isn't punished for our infra bug.
            import traceback as _tb
            tb = _tb.format_exc()
            err_type, err_msg = type(e).__name__, str(e)
            print(f"[multi_agent] agent.run crashed "
                  f"(slot={speaker_slot} role={agent.role} "
                  f"cycle={cycle} turn={turn_idx}): "
                  f"{err_type}: {err_msg}\n{tb}", flush=True)
            from .agent import AgentTurnOutput
            turn_out = AgentTurnOutput(text="")
            turn_out.worker_error = True
            turn_out.skipped_protocol_fail = True
            turn_out.skip_cause = "worker_error"
            run_error_payload = {
                "kind": "agent_run_uncaught",
                "type": err_type,
                "message": err_msg,
                "traceback": tb,
                "slot": speaker_slot,
                "role": agent.role,
                "cycle": cycle,
                "turn": turn_idx,
            }

        # Snapshot TLS counter (covers happy + crash paths) BEFORE the
        # worker thread can die and GC the dict. _commit_turn merges it
        # into the trace under the trace lock.
        turn_out.search_stub_snapshot = snapshot_search_stub_counts()
        if run_error_payload is not None:
            turn_out.run_error = run_error_payload

        # Resolve termination flag → skip_kind. A turn cannot be both
        # submit_implicit and skipped_*; see AgentTurnOutput.
        if turn_out.submit_implicit:
            skip_kind = ""
        elif turn_out.skipped_explicit:
            skip_kind = "explicit"
        else:
            skip_kind = "protocol_fail"

        am = AgentMessage(
            slot=speaker_slot,
            role=agent.role,
            cycle=cycle,
            turn=turn_idx,
            text=turn_out.text,
            n_steps=turn_out.n_steps,
            n_tool_calls=turn_out.n_tool_calls,
            n_real_tool_calls=turn_out.n_real_tool_calls,
            n_input_tokens=turn_out.n_input_tokens,
            n_output_tokens=turn_out.n_output_tokens,
            hit_step_cap=turn_out.hit_cap,
            skipped=bool(skip_kind),
            skip_kind=skip_kind,
        )
        return am, turn_out

    @staticmethod
    def _commit_turn(
        trace: "ExecutionTrace",
        lock: "threading.Lock",
        am: "AgentMessage",
        turn_out: "AgentTurnOutput",
        agent: "Agent",
        speaker_slot: int,
        arch: "ConcreteArch",
    ) -> None:
        """Apply per-turn mutations to `trace` under `lock`.

        Skipped turns (explicit or protocol_fail) are appended with text=""
        + skipped=True; the incoming filter drops them so downstream agents
        never see them. This keeps engineering failures invisible to GRPO's
        correctness signal — the architecture is judged on what active
        agents actually contribute, not on protocol noise.
        """
        skip_kind = am.skip_kind
        # protocol_compliance per role (no per-model dimension now that
        # multi-vendor dispatch is gone).
        pc_key = agent.role
        with lock:
            # Insert into trace.messages keeping (cycle, turn) ascending
            # order so downstream consumers (heuristic_extract, Synth
            # transcript, details.jsonl) see a DETERMINISTIC sequence even
            # when parallel waves commit out of completion order. Linear
            # scan from end is O(1) for the cross-wave / cross-cycle
            # append case (current cycle ≥ all prior cycles); within an
            # out-of-order wave it's O(W) where W = wave width (≤ 6).
            pos_key = (am.cycle, am.turn)
            msgs = trace.messages
            insert_idx = len(msgs)
            for i in range(len(msgs) - 1, -1, -1):
                if (msgs[i].cycle, msgs[i].turn) <= pos_key:
                    insert_idx = i + 1
                    break
                insert_idx = i
            msgs.insert(insert_idx, am)
            if am.skipped:
                trace.n_skipped_turns += 1
                if skip_kind == "protocol_fail":
                    trace.n_protocol_fail_turns += 1

            # protocol_compliance per (role, model_id)
            pc = trace.protocol_compliance.setdefault(
                pc_key,
                {
                    "submit_implicit": 0,
                    "skipped_explicit": 0,
                    "skipped_protocol_fail": 0,
                    "zero_real_tool_submits": 0,
                },
            )
            if turn_out.submit_implicit:
                pc["submit_implicit"] += 1
                if turn_out.n_real_tool_calls == 0:
                    pc["zero_real_tool_submits"] += 1
            elif turn_out.skipped_explicit:
                pc["skipped_explicit"] += 1
            else:
                pc["skipped_protocol_fail"] += 1

            # 6-way termination breakdown (categorize protocol_fail by skip_cause).
            if turn_out.submit_implicit:
                trace.termination_breakdown["submit_implicit"] += 1
            elif turn_out.skipped_explicit:
                trace.termination_breakdown["skipped_explicit"] += 1
            else:
                cause = getattr(turn_out, "skip_cause", "") or "empty_text"
                key = f"skip_{cause}"
                if key in trace.termination_breakdown:
                    trace.termination_breakdown[key] += 1
                else:
                    trace.termination_breakdown["skip_empty_text"] += 1

            trace.n_llm_calls += turn_out.n_steps
            trace.total_input_tokens += turn_out.n_input_tokens
            trace.total_output_tokens += turn_out.n_output_tokens
            if turn_out.hit_cap:
                trace.n_agents_hit_step_cap += 1
            trace.total_tool_calls += turn_out.n_tool_calls
            for tname, args, tout, elapsed_s in turn_out.tool_log:
                trace.tool_call_counts[tname] = (
                    trace.tool_call_counts.get(tname, 0) + 1
                )
                # Per-tool duration log for forensics on data-driven timeout
                # tuning. Snippet is bounded to 400 chars to keep details.jsonl
                # within reasonable size; `ok` flags whether the tool output
                # contained an engine-error sentinel.
                ok = not (f"[{tname}] TIMEOUT" in (tout or "")
                          or f"[{tname}] ERROR" in (tout or ""))
                if tname == "python_exec":
                    snip = (args or "")[:800]   # see python_exec_log docstring
                    trace.python_exec_log.append((float(elapsed_s), snip, ok))
                # Classify real tool failures into err_kind. We do NOT count
                # pytest "FAILED test_xxx" or python_exec stderr "Error:" —
                # those are the model's code being wrong, not the tool.
                # Two sentinel formats accepted:
                #   `[tname] TIMEOUT/ERROR`         compute tools
                #   `[tname: ErrClass: msg]`        Serper-backed tools
                # The NOT_INSTALLED branch must come BEFORE the generic
                # ERROR branch because pytest_runner emits
                # `[pytest_runner] ERROR pytest not installed`.
                tout_l = tout or ""
                err_kind = ""
                if f"[{tname}] TIMEOUT" in tout_l:
                    err_kind = "TIMEOUT"
                elif tname == "pytest_runner" and "pytest not installed" in tout_l:
                    err_kind = "NOT_INSTALLED"
                elif f"[{tname}] ERROR" in tout_l:
                    err_kind = "ERROR"
                elif "not available for this role" in tout_l:
                    err_kind = "ROLE_MISMATCH"
                elif tout_l.startswith(f"[{tname}: "):
                    # `[web_search: HTTPError: ...]` — classify by the
                    # embedded exception class for the forensic histogram.
                    import re as _re
                    m = _re.match(rf"\[{_re.escape(tname)}:\s*([^:\]]+)",
                                  tout_l)
                    err_kind = (m.group(1).strip() if m else "STUB").upper()
                if err_kind:
                    trace.tool_error_counts[tname] = (
                        trace.tool_error_counts.get(tname, 0) + 1
                    )
                    ek = f"{tname}:{err_kind}"
                    trace.tool_error_kinds[ek] = (
                        trace.tool_error_kinds.get(ek, 0) + 1
                    )
                if "[truncated]" in tout:
                    trace.n_tool_truncations += 1
            if turn_out.worker_error:
                # Both routes set worker_error=True (so the trace-level
                # skip path / eng_valid both fire) but they're semantically
                # different — split them to keep dashboards honest.
                #   skip_cause == "truncated": output cap hit (verbose
                #     model, finish_reason='length'). Routine.
                #   else: 6-retry-exhausted chat / agent.run crash. Real
                #     infra noise — what api_error_high should alert on.
                if turn_out.skip_cause == "truncated":
                    trace.n_worker_truncations += 1
                else:
                    trace.n_api_errors += 1

            # Merge stub counts from this turn's worker thread into the
            # trace. Doing it here (under the trace lock) means the
            # numbers survive across inner ThreadPool waves; previously
            # the main thread snapshotted its OWN TLS which never got
            # touched by inner workers, so default `parallel_within_cycle`
            # traces had `search_stub_counts = {0,0,0}` regardless of how
            # many real stubs the agents accumulated.
            # Merge per-turn TLS snapshots into the trace (under the
            # trace lock). Without this, ThreadPoolExecutor worker
            # threads' TLS dicts get GC'd on thread exit and the
            # forensic telemetry is lost.
            snap = getattr(turn_out, "search_stub_snapshot", None)
            if snap:
                for k, v in snap.items():
                    trace.search_stub_counts[k] = (
                        trace.search_stub_counts.get(k, 0) + int(v)
                    )

            # Forward structured run-error (agent.run crash) onto the
            # trace so details.jsonl keeps full traceback + slot/role
            # attribution — never lost to a stderr printout alone.
            run_err = getattr(turn_out, "run_error", None)
            if run_err:
                trace.run_errors.append(run_err)

    # ------------------------------------------------------------------
    def _worker_for(self, arch: "ConcreteArch", slot: int) -> "Worker":
        """Resolve the worker for an agent slot. Single-model (no pool /
        no model assignment) → the shared worker; multi-model → the
        head-chosen model from the pool."""
        if self.worker_pool is None or arch.model is None:
            return self.worker
        model_name = self.spec.model_names[int(arch.model[slot].item())]
        return self.worker_pool[model_name]

    def run(self, task: str, arch: ConcreteArch) -> ExecutionTrace:
        # Deferred imports to avoid circular import at module load time
        import time as _time
        from .agent import Agent
        from .synth import Synth, heuristic_extract

        spec = self.spec
        trace = ExecutionTrace(task=task, arch=arch)
        trace_lock = threading.Lock()
        t_start = _time.time()
        wall_deadline = t_start + self.wall_clock_timeout_s if self.wall_clock_timeout_s > 0 else None

        # Stub-count attribution: every _run_one_turn resets its own
        # worker-thread TLS at entry and snapshots at exit, attaching
        # the result to turn_out for _commit_turn to merge into
        # trace.search_stub_counts. The main thread itself never does
        # web_search, so no reset/snapshot needed here.

        # Build one Agent per active slot. Single-model mode → every slot
        # uses self.worker. Multi-model mode (worker_pool set + arch carries
        # a model assignment) → slot s runs on the head-chosen model.
        active_idx = arch.sequence.tolist()
        agents: dict[int, Agent] = {}
        for slot in active_idx:
            role = spec.role_names[int(arch.roles[slot].item())]
            agents[slot] = Agent(
                slot=slot,
                role=role,
                worker=self._worker_for(arch, slot),
                max_steps=spec.safety_max_steps,
                max_new_tokens=self.max_new_tokens,
            )

        synth = Synth(self.synth_worker, max_new_tokens=self.synth_max_new_tokens)

        # Wave decomposition is per-cycle but identical across cycles (depends
        # only on arch.edges + active_idx). Compute once.
        if self.parallel_within_cycle:
            waves = self._compute_waves(arch, active_idx)
        else:
            waves = [[s] for s in active_idx]   # serial fallback
        # turn_idx for each speaker = its position in original active_idx
        # (preserved across waves for prompt consistency)
        turn_idx_of: dict[int, int] = {s: i for i, s in enumerate(active_idx)}

        wall_clock_hit = False
        call_cap_hit = False
        for cycle in range(spec.safety_max_cycles):
            if wall_deadline is not None and _time.time() > wall_deadline:
                wall_clock_hit = True
                break
            if self.max_llm_calls_per_trace > 0 and trace.n_llm_calls >= self.max_llm_calls_per_trace:
                call_cap_hit = True
                break
            # Iterate waves. Within a wave, all agents have no edge between
            # them so they can run concurrently with identical semantics to
            # any sequential order.
            for wave in waves:
                if wall_deadline is not None and _time.time() > wall_deadline:
                    wall_clock_hit = True
                    break
                if self.max_llm_calls_per_trace > 0 and trace.n_llm_calls >= self.max_llm_calls_per_trace:
                    call_cap_hit = True
                    break

                # Snapshot trace.messages ONCE for the wave (all wave agents
                # see the same prior state — they're independent per the DAG).
                with trace_lock:
                    msgs_snapshot = list(trace.messages)
                wave_incomings = {
                    s: self._build_incoming(msgs_snapshot, s, arch,
                                            cycle, self.history_cycles)
                    for s in wave
                }

                if len(wave) == 1:
                    # Single-agent wave: run inline (no thread overhead).
                    s = wave[0]
                    am, turn_out = self._run_one_turn(
                        agent=agents[s], task=task,
                        incoming=wave_incomings[s],
                        cycle=cycle, turn_idx=turn_idx_of[s],
                        n_turns=len(active_idx),
                        wall_deadline=wall_deadline,
                        speaker_slot=s,
                    )
                    self._commit_turn(trace, trace_lock, am, turn_out,
                                      agents[s], s, arch)
                else:
                    # Multi-agent wave: parallel via inner thread pool. The
                    # pool size matches wave width so we don't oversubscribe
                    # API; tracking only the outer (per-trace) max
                    # concurrency in grpo._run_one is the broader budget.
                    with ThreadPoolExecutor(max_workers=len(wave)) as wpool:
                        futs = {
                            wpool.submit(
                                self._run_one_turn,
                                agents[s], task, wave_incomings[s],
                                cycle, turn_idx_of[s], len(active_idx),
                                wall_deadline, s,
                            ): s
                            for s in wave
                        }
                        for fut in as_completed(futs):
                            slot_done = futs[fut]
                            am, turn_out = fut.result()
                            self._commit_turn(trace, trace_lock, am, turn_out,
                                              agents[slot_done], slot_done, arch)

            trace.n_cycles_run = cycle + 1
            if wall_clock_hit or call_cap_hit:
                break

            # Synth check after each cycle
            transcript_items = [
                (m.slot, m.role, m.cycle, m.text) for m in trace.messages
            ]
            try:
                verdict = synth.judge(task, transcript_items)
            except Exception as e:  # noqa: BLE001
                # Synth crash → keep iterating cycles (treat as "no verdict
                # yet"); the safety_max_cycles fallback to heuristic_extract
                # will handle final answer.
                import traceback as _tb
                tb = _tb.format_exc()
                err_type, err_msg = type(e).__name__, str(e)
                print(f"[multi_agent] synth.judge crashed (cycle={cycle}): "
                      f"{err_type}: {err_msg}\n{tb}", flush=True)
                from .synth import SynthVerdict
                verdict = SynthVerdict(is_done=False, malformed=True,
                                       raw_output=f"[synth crash: {err_type}]")
                trace.n_api_errors += 1
                # Structured record — without this, synth crashes in the
                # outer GRPO ThreadPoolExecutor leave only interleaved
                # stderr that's hard to attribute back to a trace.
                trace.run_errors.append({
                    "kind": "synth_crash",
                    "type": err_type,
                    "message": err_msg,
                    "traceback": tb,
                    "cycle": cycle,
                })
            trace.n_llm_calls += 1
            trace.n_synth_calls += 1
            trace.total_input_tokens += verdict.n_input_tokens
            trace.total_output_tokens += verdict.n_output_tokens
            trace.synth_log.append(verdict.raw_output)
            if verdict.is_done:
                # For code answers, Synth's terse "ANSWER: X" loses the body.
                # Check the transcript for a code-block Candidate/Verified/Refined
                # marker; if found, prefer it over Synth's short string.
                items = [(m.slot, m.role, m.cycle, m.text) for m in trace.messages]
                code_answer = heuristic_extract(items)
                if code_answer.startswith("```python"):
                    trace.final_answer = code_answer
                else:
                    trace.final_answer = verdict.answer
                trace.final_via_synth = True
                # NB: trace.search_stub_counts is already populated by
                # _commit_turn merging each turn's snapshot under the
                # trace lock (works across inner-pool waves).
                trace.wall_seconds = _time.time() - t_start
                return trace

        # Hit safety cap → heuristic fallback. Exactly one of the three
        # cap-hit flags below is True: hit_wall_clock / hit_call_cap /
        # hit_cycle_cap. They're mutually exclusive so telemetry can attribute
        # the cap precisely (previously hit_cycle_cap was set on every fallback
        # path, masking which cap actually fired).
        transcript_items = [(m.slot, m.role, m.cycle, m.text) for m in trace.messages]
        trace.final_answer = heuristic_extract(transcript_items)
        trace.final_via_synth = False
        if wall_clock_hit:
            # ARCHITECTURE chose a path that ran over wall clock. Reward is
            # NOT masked: correctness of heuristic_extract drives the
            # advantage directly. Architectures that often cap out will
            # naturally accumulate lower reward.
            trace.hit_wall_clock = True
            trace.n_arch_caps_hit += 1
        elif call_cap_hit:
            trace.hit_call_cap = True
            trace.n_arch_caps_hit += 1
        else:
            # Synth said CONTINUE for safety_max_cycles in a row — the
            # arch's chosen "iterate until done" loop didn't converge. Same
            # attribution as wall_clock / call_cap (arch chose the iterate-
            # forever pattern), so we count it under n_arch_caps_hit too.
            trace.hit_cycle_cap = True
            trace.n_arch_caps_hit += 1
        # trace.search_stub_counts is populated incrementally by _commit_turn
        # — see the matching comment in the synth-DONE branch above.
        trace.wall_seconds = _time.time() - t_start
        return trace


__all__ = [
    "AgentMessage",
    "ExecutionTrace",
    "MockWorker",
    "MultiAgentExecutor",
    "Worker",
    "WorkerOutput",
]
