"""Evaluate a model (head or baseline) on a benchmark.

Usage examples
--------------

# Baseline: solver-verifier on GSM8K (200 questions)
python scripts/04_evaluate.py \
    --mode baseline --baseline solver_verifier \
    --dataset gsm8k --n 200 \
    --worker gpugeek --worker_model Vendor3/DeepSeek-V4-Pro \
    --out_jsonl results/sv_gsm8k.jsonl

# Our trained head on GSM8K (deterministic mode)
python scripts/04_evaluate.py \
    --mode head --head_ckpt checkpoints/sft_v1/head_step100 \
    --dataset gsm8k --n 200 \
    --worker gpugeek --worker_model Vendor3/DeepSeek-V4-Pro \
    --out_jsonl results/apo_gsm8k.jsonl

# Self-consistency over architectures (sample 5 archs, vote)
python scripts/04_evaluate.py --mode head --head_ckpt ... \
    --self_consistency 5 --head_sample
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

# Make sibling `_common.py` importable regardless of how this file was
# loaded (direct execution puts scripts/ on sys.path automatically;
# spec_from_file_location does not).
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Approximate prices (USD / 1K tokens, cache-miss list). Used for cost
# rollup in eval reports.
TOKEN_PRICES = {
    # DeepSeek native
    "deepseek-v4-flash": (0.00014, 0.00028),
    "deepseek-v4-pro":   (0.00174, 0.00348),
    # Qwen via Aliyun Bailian (¥ → $ at ~7.0)
    "qwen-flash":        (0.000029, 0.00021),
    "qwen3.5-plus":      (0.00011,  0.00069),
    "qwen3.6-plus":      (0.00029,  0.00171),
    "qwen3.7-max":       (0.00086,  0.00257),
    # GpuGeek catalogue
    "Vendor3/DeepSeek-V4-Pro":   (0.00174, 0.00348),
    "Vendor3/DeepSeek-V4-Flash": (0.00014, 0.00028),
    "Vendor2/GPT-5.5":           (0.00125, 0.01000),
    "Vendor2/GPT-5.1":           (0.00125, 0.01000),  # judge in HLE v1
    "Vendor2/Claude-4.7-opus":   (0.01500, 0.07500),
    "Vendor2/Gemini-3.1-pro":    (0.00125, 0.01000),
}
# Mid-tier fallback for unknown ids — conservative over-count beats a
# silent $0. Mid-tier (DeepSeek-Pro) is roughly the median of our pool.
_FALLBACK_PRICE = (0.00174, 0.00348)
_WARNED_UNKNOWN_PRICE: set[str] = set()


def _price(model: str, n_in: int, n_out: int) -> float:
    if model in TOKEN_PRICES:
        p_in, p_out = TOKEN_PRICES[model]
    else:
        p_in, p_out = _FALLBACK_PRICE
        if model not in _WARNED_UNKNOWN_PRICE:
            _WARNED_UNKNOWN_PRICE.add(model)
            print(f"[_price] WARN model={model!r} not in TOKEN_PRICES; "
                  f"using mid-tier fallback ${p_in*1000:.3f}/${p_out*1000:.3f} "
                  f"per 1K tok. Add it to TOKEN_PRICES for accurate cost.",
                  flush=True)
    return n_in / 1000 * p_in + n_out / 1000 * p_out


from _common import build_judge as _build_judge  # noqa: E402
from _common import build_worker as _build_worker  # noqa: E402


def _build_archs_for_task(args, task_text, head=None, tokenizer=None,
                          device=None, arch_spec=None):
    """Either return a fixed-baseline arch, or sample one (or many) from head."""
    if args.mode == "baseline":
        from arch_policy.baselines import get_baseline
        return [get_baseline(args.baseline)]

    # mode == "head"
    import torch
    from arch_policy import ARCH, sample_arch, to_arch_logits
    spec = arch_spec or ARCH

    enc = tokenizer([task_text], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = head(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    logits = to_arch_logits(out, batch_idx=0)
    # Move to CPU for sampling (uses Python lists internally). Keep
    # model_logits (multi-model heads) so the model dimension is sampled.
    logits_cpu = type(logits)(
        gate_logits=logits.gate_logits.cpu(),
        role_logits=logits.role_logits.cpu(),
        edge_logits=logits.edge_logits.cpu(),
        seq_scores =logits.seq_scores.cpu(),
        model_logits=(logits.model_logits.cpu()
                      if logits.model_logits is not None else None),
    )
    n_samples = max(1, args.self_consistency)
    archs = []
    for _ in range(n_samples):
        archs.append(sample_arch(logits_cpu, spec,
                                 deterministic=not args.head_sample))
    return archs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "head"], required=True)
    ap.add_argument("--baseline", default="single",
                    help="baseline name (see arch_policy.baselines.BASELINE_REGISTRY)")
    ap.add_argument("--head_ckpt", default=None, help="path to head checkpoint dir")
    ap.add_argument("--head_model", default="Qwen/Qwen3-4B")
    ap.add_argument("--lora_rank", type=int, default=0,
                    help="LoRA rank (must match SFT setting if non-zero).")
    ap.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--head_device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    ap.add_argument("--head_sample", action="store_true",
                    help="stochastic sampling instead of deterministic (needed for self_consistency > 1)")
    ap.add_argument("--self_consistency", type=int, default=1,
                    help="how many architectures to sample per task (head mode only)")

    src = ap.add_mutually_exclusive_group()
    src.add_argument(
        "--bench", default=None,
        help="Per-bench plugin id. When set: dataset from adapter, "
             "grading from adapter.grade() via the LLM judge.",
    )
    src.add_argument(
        "--dataset", default=None,
        choices=["synthetic",
                 # classical
                 "gsm8k", "math", "humaneval", "mbpp", "mmlu", "bbh", "arc",
                 # agent-system benches
                 "browsecomp", "hle", "phybench", "livecodebench"],
        help="Legacy --dataset path (rule grader). 'gsm8k' default when "
             "neither --bench nor --dataset is given.",
    )
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42,
                    help="MUST match the training seed (42 = production): "
                         "the bench train/test split is seed-determined, so "
                         "eval at a different seed would draw a test set that "
                         "OVERLAPS the training split. Leave at 42.")

    # LLM judge (used by --bench adapters that need one).
    ap.add_argument("--judge", choices=["mock", "gpugeek", "deepseek", "qwen"],
                    default="gpugeek")
    ap.add_argument("--judge_model", default="Vendor2/GPT-5.1",
                    help="Concrete model id (default Vendor2/GPT-5.1 = "
                         "production judge). Pass '' to use the rule grader "
                         "(exec-graded benches need no judge).")
    ap.add_argument("--judge_timeout", type=float, default=90.0)

    ap.add_argument("--worker", choices=["mock", "gpugeek", "deepseek", "qwen"],
                    default="qwen",
                    help="Worker backend. Match the worker used during "
                         "training for fair comparison. Default `qwen` "
                         "(matches HLE v1 production training; intern "
                         "audit #26 — previously defaulted to deepseek "
                         "which was a quiet footgun if user forgot the "
                         "override).")
    ap.add_argument("--worker_model", default="qwen3.7-max",
                    help="LLM identifier for the chosen --worker backend "
                         "(single-model mode).")
    ap.add_argument("--worker_models", default=None,
                    help="Comma-separated model pool — REQUIRED to eval a "
                         "head trained with the per-agent model dimension "
                         "(its ckpt has head_M). Must match the training "
                         "pool, e.g. 'qwen3.7-max,kimi-k2.6,glm-5.1,"
                         "deepseek-v4-pro'. Omit for single-model heads.")
    ap.add_argument("--worker_temperature", type=float, default=0.0)
    ap.add_argument("--worker_timeout", type=float, default=600.0,
                    help="Per-call timeout (s). 600 = production (a "
                         "saturating max_new_tokens=8192 reply ≈112s).")
    ap.add_argument("--worker_thinking", action=argparse.BooleanOptionalAction,
                    default=False)
    ap.add_argument("--mock_answer", default="42")

    ap.add_argument("--max_new_tokens", type=int, default=8192,
                    help="Max generated tokens per worker LLM call (ArchSpec.safety_max_tokens_per_call)")
    ap.add_argument("--history_cycles", type=int, default=0,
                    help="Cap past cycles in each agent's incoming prompt slice "
                         "(0 = no limit, default).")

    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--verbose", action="store_true")

    ap.add_argument(
        "--strict_tools", action=argparse.BooleanOptionalAction, default=True,
        help="Fail loudly at startup if SERPER_API_KEY is missing (default "
             "ON). Same contract as 03_train_grpo's flag: without a real "
             "key the search tools return offline-stub strings, which would "
             "depress eval scores on BrowseComp / HLE / arc-research and "
             "you'd blame the head. Pass --no-strict_tools to acknowledge "
             "the risk (every trace's `search_stub_counts` will still flag it).",
    )

    args = ap.parse_args()

    # --- Argument cross-validation -------------------------------------
    # `--self_consistency N>1` is pointless without `--head_sample`:
    # deterministic sampling yields the same arch every time, so N copies
    # of the same trace would just waste API budget. Fail loud.
    if args.self_consistency > 1 and not args.head_sample:
        raise SystemExit(
            f"--self_consistency={args.self_consistency} requires "
            "--head_sample (stochastic sampling). Without it the head "
            "produces the same architecture every time and you'd just pay "
            "for N identical traces. Either pass --head_sample or set "
            "--self_consistency 1."
        )

    from arch_policy import MultiAgentExecutor, bench, seed_all
    from arch_policy.data.tasks import load_huggingface, load_local_synthetic
    from arch_policy.executor.tools import preflight_tools
    from arch_policy.reward import grade

    if args.strict_tools:
        preflight_tools()
    else:
        print("[eval] --no-strict_tools: search tools may degrade silently; "
              "watch per-trace search_stub_counts in the output JSONL.",
              flush=True)

    seed_all(args.seed)

    # --- Load tasks ---
    adapter = None
    judge_worker = None
    if args.bench is not None:
        adapter = bench.get(args.bench)
        tasks = adapter.load_split(args.split, seed=args.seed)
        if args.n and args.n < len(tasks):
            tasks = tasks[: args.n]
        # Build the judge only for benches that grade with one — exec-graded
        # benches (LiveCodeBench) skip it so the default judge_model never
        # forces an unused GpuGeek key.
        judge_worker = _build_judge(args) if adapter.needs_judge() else None
        if adapter.needs_judge() and judge_worker is None:
            print(f"[eval] WARN: bench={args.bench} wants a judge but "
                  f"--judge_model unset — using rule-grader fallback.")
        print(f"[eval] bench={args.bench} loaded {len(tasks)} "
              f"{args.split} tasks; judge="
              f"{args.judge_model if adapter.needs_judge() else 'n/a (exec-graded)'}")
    elif (args.dataset or "gsm8k") == "synthetic":
        tasks = load_local_synthetic(n_per_family=max(1, args.n // 3), seed=args.seed)[: args.n]
        print(f"[eval] loaded {len(tasks)} synthetic tasks")
    else:
        dataset = args.dataset or "gsm8k"
        tasks = load_huggingface(dataset, split=args.split, n=args.n, seed=args.seed)
        print(f"[eval] loaded {len(tasks)} tasks from {dataset}/{args.split}")

    # --- ArchSpec (+ optional multi-model pool — must match training) ---
    from dataclasses import replace as _replace
    from arch_policy import ARCH
    arch_spec = ARCH
    worker_pool = None
    if args.worker_models:
        model_names = tuple(m.strip() for m in args.worker_models.split(",") if m.strip())
        arch_spec = _replace(ARCH, model_names=model_names)
        from _common import build_worker_pool
        worker_pool = build_worker_pool(
            model_names, timeout=args.worker_timeout,
            temperature=args.worker_temperature, thinking=args.worker_thinking)
        print(f"[eval] model pool ({arch_spec.n_models}): "
              f"{', '.join(model_names)} — per-agent model dimension ACTIVE")

    # --- Build executor ---
    worker = (worker_pool[arch_spec.model_names[0]] if worker_pool
              else _build_worker(args))
    print(f"[eval] worker={args.worker}={args.worker_model}")
    executor = MultiAgentExecutor(
        worker=worker,
        spec=arch_spec,
        worker_pool=worker_pool,
        max_new_tokens_per_call=args.max_new_tokens,
        history_cycles=args.history_cycles,
    )

    # --- Build head if needed ---
    head = None
    tokenizer = None
    if args.mode == "head":
        if args.head_ckpt is None:
            raise ValueError("--head_ckpt required in head mode")
        from arch_policy import ArchitectureHead, load_tokenizer
        from arch_policy.training.sft import load_head_checkpoint
        tokenizer = load_tokenizer(args.head_model)
        head = ArchitectureHead(
            backbone_name=args.head_model,
            arch_spec=arch_spec,   # n_models>1 → builds head_M to match ckpt
            freeze_backbone=args.freeze_backbone,
            lora_rank=args.lora_rank,
            torch_dtype="float32" if args.head_device == "cpu" else "bfloat16",
        )
        load_head_checkpoint(head, args.head_ckpt)
        head.to(args.head_device).eval()
        print(f"[eval] loaded head from {args.head_ckpt} "
              f"(freeze_backbone={args.freeze_backbone}, lora_rank={args.lora_rank}, "
              f"n_models={arch_spec.n_models})")

    # --- Run ---
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("w")

    n_correct = 0.0
    total_in = 0
    total_out = 0
    total_calls = 0
    total_cost = 0.0
    t0 = time.time()

    for i, sample in enumerate(tasks):
        archs = _build_archs_for_task(
            args, sample.task,
            head=head, tokenizer=tokenizer, device=args.head_device,
            arch_spec=arch_spec,
        )
        votes: Counter = Counter()
        per_arch_outputs = []
        for arch in archs:
            trace = executor.run(sample.task, arch)
            per_arch_outputs.append(trace)
            ans = trace.final_answer
            votes[ans] += 1
            total_in += trace.total_input_tokens
            total_out += trace.total_output_tokens
            total_calls += trace.n_llm_calls
            total_cost += _price(args.worker_model, trace.total_input_tokens, trace.total_output_tokens)
        if not votes:
            final_answer = ""
        else:
            final_answer = votes.most_common(1)[0][0]
        if adapter is not None:
            score, judge_audit = adapter.grade(final_answer, sample, judge=judge_worker)
            # Judge cost — priced separately via judge_model (different
            # vendor from worker). bench adapters expose token counts
            # in audit['judge_in_tokens'] / audit['judge_out_tokens'].
            if isinstance(judge_audit, dict):
                j_in = int(judge_audit.get("judge_in_tokens", 0) or 0)
                j_out = int(judge_audit.get("judge_out_tokens", 0) or 0)
                if (j_in or j_out) and args.judge_model:
                    total_cost += _price(args.judge_model, j_in, j_out)
                    total_in += j_in   # surface in headline cost rollup
                    total_out += j_out
        else:
            score, judge_audit = grade(final_answer, sample), None
        n_correct += score

        # Silent-degradation watch: aggregate stub counts + run-error
        # types across the architectures voted in for this task. Non-zero
        # `search_stub_total` means Serper returned stubs for some calls
        # (HTTP errors or — if --no-strict_tools — missing key); this
        # corrupts BrowseComp/HLE accuracy. `run_errors_total` is the
        # count of structured infra failures (agent.run/synth crashes).
        stub_per_arch = [
            sum((getattr(t, "search_stub_counts", {}) or {}).values())
            for t in per_arch_outputs
        ]
        run_err_per_arch = [
            len(getattr(t, "run_errors", []) or []) for t in per_arch_outputs
        ]
        # Sorted role names per arch — groups equivalent architectures
        # across slot permutations for downstream rollups in 06_analyze_bench.
        role_multisets = [
            tuple(sorted(t.arch.role_name(i) for i in t.arch.sequence.tolist()))
            for t in per_arch_outputs
        ]
        rec = {
            "task_id": sample.task_id,
            "family": sample.family,
            "subject": sample.metadata.get("subject") if sample.metadata else None,
            "gold": sample.gold_answer,
            "prediction": final_answer,
            "score": score,
            "judge_audit": judge_audit,
            "n_archs_voted": len(archs),
            "tokens_in": sum(t.total_input_tokens for t in per_arch_outputs),
            "tokens_out": sum(t.total_output_tokens for t in per_arch_outputs),
            "llm_calls": sum(t.n_llm_calls for t in per_arch_outputs),
            "n_cycles": [t.n_cycles_run for t in per_arch_outputs],
            "n_synth_calls": [t.n_synth_calls for t in per_arch_outputs],
            "via_synth": [t.final_via_synth for t in per_arch_outputs],
            "arch_summary": [t.arch.to_summary() for t in per_arch_outputs],
            "role_multisets": role_multisets,
            "n_active_per_arch": [int(t.arch.active_mask.sum().item()) for t in per_arch_outputs],
            "n_edges_per_arch": [int(t.arch.edges.sum().item()) for t in per_arch_outputs],
            "search_stub_per_arch": stub_per_arch,
            "search_stub_total": sum(stub_per_arch),
            "run_errors_per_arch": run_err_per_arch,
            "run_errors_total": sum(run_err_per_arch),
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()

        if args.verbose or (i + 1) % 10 == 0:
            print(
                f"[eval] {i+1}/{len(tasks)}  acc={n_correct/(i+1):.3f}  "
                f"avg_calls={total_calls/(i+1):.1f}  cost=${total_cost:.3f}",
                flush=True,
            )

    fout.close()
    elapsed = time.time() - t0
    n = len(tasks)
    print("\n=== Summary ===")
    print(f"  accuracy:        {n_correct / n:.4f}  ({int(n_correct)}/{n})")
    print(f"  avg tokens in:   {total_in / n:.1f}")
    print(f"  avg tokens out:  {total_out / n:.1f}")
    print(f"  avg LLM calls:   {total_calls / n:.2f}")
    print(f"  total cost USD:  ${total_cost:.3f}")
    print(f"  wall time:       {elapsed:.1f}s")
    print(f"  results saved → {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
