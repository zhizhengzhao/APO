"""Architecture-level GRPO fine-tuning runner.

Samples G architectures per task, runs them through the executor, scores
(correct ∈ {0,1} + n_calls), and updates the head via
`training.grpo.shaped_advantage` (correctness sign + cost bonus + /σ).

Two starting modes — choose by whether you pass `--head_ckpt`:

  - From-scratch (DEFAULT, no --head_ckpt): ArchitectureHead's default
    random init for all typed heads. Inject curriculum (family-stratified
    canonical archs) + entropy bonus guide early exploration. Recommended
    combo: --freeze_backbone so only ~1M typed-head params train.

  - SFT warm-start (--head_ckpt PATH): Optional — loads heads that have
    already been pretrained on the canonical-arch library.

Canonical command (cheapest, from-scratch + inject curriculum):

    python scripts/03_train_grpo.py \\
        --freeze_backbone \\
        --dataset mixed_stress --n 1000 --epochs 3 \\
        --inject_mode family_stratified --inject_k_schedule "8,4,2" \\
        --out_dir checkpoints/grpo_v1
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

# Make sibling `_common.py` importable even when this file is loaded
# via spec_from_file_location (which doesn't auto-add the script's
# directory to sys.path the way `python scripts/03_train_grpo.py` does).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import build_judge as _build_judge  # noqa: E402
from _common import build_worker as _build_worker  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head_ckpt", default=None,
                    help="OPTIONAL path to SFT-warmed head checkpoint. If "
                         "omitted (default), GRPO starts from-scratch with "
                         "random-init typed heads — inject curriculum + "
                         "entropy bonus drive early exploration. Recommended "
                         "with --freeze_backbone for fast convergence.")
    ap.add_argument("--head_model", default="Qwen/Qwen3-4B")
    ap.add_argument("--head_dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--head_device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=False,
                    help="If set, only typed heads are trained. "
                         "Match the SFT-stage setting for correct ckpt loading.")
    ap.add_argument("--lora_rank", type=int, default=32,
                    help="LoRA rank (production: 32 — must match the SFT "
                         "stage for warm-start ckpt compatibility). Pass 0 "
                         "for full-FT or use --freeze_backbone for heads-only.")
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Trade compute for activation memory (production: "
                         "ON — required to fit LoRA-32 seq=1024 on one 80GB "
                         "card). Pass --no-gradient_checkpointing to disable.")

    # Per-bench plugin OR legacy --dataset. Plugin wins when both set.
    src = ap.add_mutually_exclusive_group()
    src.add_argument(
        "--bench", default=None,
        help="Per-bench plugin id (one file per bench under bench/). "
             "When set: dataset comes from adapter.load_split('train'), "
             "reward comes from adapter.grade() via the LLM judge, and "
             "inject is forced OFF (SFT prior IS the architectural prior).",
    )
    src.add_argument(
        "--dataset", default=None,
        choices=["synthetic",
                 # classical
                 "gsm8k", "math", "humaneval", "mbpp", "mmlu", "bbh", "arc",
                 # graduate STEM (GPQA-Diamond mirror)
                 "gpqa",
                 # agent-system benches
                 "browsecomp", "hle", "phybench", "livecodebench",
                 # stress mix: BrowseComp 30% + HLE 30% + GPQA 15% +
                 # PHYBench 10% + MATH-500-L5 5% + LiveCodeBench 10%
                 "mixed_stress"],
        help="Legacy --dataset path (rule-based grading). Default 'gsm8k' "
             "when neither --bench nor --dataset is set.",
    )
    ap.add_argument("--split", default="train")
    ap.add_argument("--train_ratio", type=float, default=0.8,
                    help="Train/holdout split ratio for the --bench loader. "
                         "1.0 = use ALL problems for training (no holdout). "
                         "Valid for APO: we train the architecture policy, not "
                         "the worker, so there is no eval-leakage concern in "
                         "training on the full benchmark.")
    ap.add_argument("--n", type=int, default=200, help="number of training tasks")

    ap.add_argument("--worker", choices=["mock", "gpugeek", "deepseek", "qwen"],
                    default="qwen",
                    help="`qwen` (default, production): Aliyun Bailian "
                         "DashScope (OpenAI-compat). "
                         "`deepseek`: DeepSeek native API. "
                         "`qwen` also serves the multi-model pool for "
                         "qwen3.7-max / qwen-flash / etc. "
                         "`gpugeek`: GpuGeek gateway for any model in their "
                         "catalogue. `mock`: deterministic test stub.")
    ap.add_argument("--worker_model", default="qwen3.7-max",
                    help="LLM identifier for the chosen --worker backend "
                         "(single-model mode). Default qwen3.7-max = "
                         "production.")
    ap.add_argument("--worker_models", default=None,
                    help="Comma-separated model pool for the per-agent "
                         "model-selection dimension (e.g. "
                         "'qwen3.7-max,kimi-k2.6,glm-5.1,deepseek-v4-pro'). "
                         "All served via DashScope. When given (>1 model), "
                         "the head's 5th typed head learns per-slot model "
                         "choice; GRPO explores it from a uniform prior. "
                         "Omit for single-model (4-head) training.")
    ap.add_argument("--worker_temperature", type=float, default=0.0)
    ap.add_argument("--worker_timeout", type=float, default=600.0,
                    help="Per-LLM-call hard timeout (seconds). With "
                         "--max_new_tokens=8192 a saturating reply was "
                         "measured at 112s (≈73 tok/s). 600s = ~5x "
                         "headroom for the slowest-tail. wall_clock_"
                         "timeout_s caps the WHOLE trace.")
    ap.add_argument("--worker_thinking", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Enable model thinking / reasoning_effort=high. "
                         "Disabled by default for speed.")
    ap.add_argument("--mock_answer", default="42")

    # LLM judge (used by --bench adapters that need one, e.g. HLE).
    ap.add_argument("--judge", choices=["mock", "gpugeek", "deepseek", "qwen"],
                    default="gpugeek",
                    help="Vendor for the judge worker. Default `gpugeek` "
                         "(routes to OpenAI/Azure-GPT-5.1 etc.).")
    ap.add_argument("--judge_model", default="Vendor2/GPT-5.1",
                    help="Concrete model id for the judge (default "
                         "Vendor2/GPT-5.1 = production HLE/FrontierScience "
                         "judge). Pass '' to disable (rule-grader fallback; "
                         "e.g. exec-graded LiveCodeBench needs no judge).")
    ap.add_argument("--judge_timeout", type=float, default=90.0,
                    help="Judge call timeout (s). 90 = production (judge "
                         "p95≈3s + rare reasoning runaway headroom).")

    ap.add_argument("--G", type=int, default=8,
                    help="GRPO group size (production: 8 — see decision log "
                         "B×G 12×12→16×8 for the speed/advantage trade-off).")
    ap.add_argument("--batch_size", type=int, default=16,
                    help="tasks per GRPO step (production: 16; B×G=128 "
                         "traces/step fits one wave at max_concurrent=128).")
    ap.add_argument("--epochs", type=int, default=5,
                    help="production: 5 epochs over the bench train split.")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--entropy_weight", type=float, default=1.0,
                    help="Global scaler over DEFAULT_ENTROPY_WEIGHTS. "
                         "Default 1.0 = use per-component weights as-is.")
    ap.add_argument("--cost_bonus_scale", type=float, default=0.5,
                    help="shaped_advantage cost-bonus scale. Within a task's "
                         "correct sub-group, cheapest gets +scale on top of "
                         "the +1 base, most expensive gets +0. Defaults to "
                         "0.5 → correct ∈ [+1, +1.5] vs wrong = -1 (before /σ): "
                         "cost is a light tiebreak, correctness stays dominant.")
    ap.add_argument("--max_seq_len", type=int, default=1024,
                    help="Max tokens for the head's tokenizer "
                         "(TrainSpec.tokenizer_max_len). 1024 covers "
                         "96.6%% of HLE tasks; truncation makes the "
                         "head score archs from a partial task.")
    ap.add_argument("--max_new_tokens", type=int, default=8192,
                    help="Max generated tokens per worker LLM call. "
                         "Set to qwen3.7-max model ceiling (8192) so "
                         "EVERY worker reply finishes naturally — never "
                         "truncated mid-reasoning before ACTION: submit. "
                         "Empirical probe (Solver, 20 HLE tasks, 2026-05-27):\n"
                         "  cap=1024 → 50%% truncation 🔴\n"
                         "  cap=2048 → 30%% truncation 🔴\n"
                         "  cap=4096 → 20%% truncation 🔴\n"
                         "  cap=8192 →  0%% truncation ✅\n"
                         "Worst-case saturating call = ~110s; mean = "
                         "2147 tokens (~24s). Truncation is a meta-"
                         "confound the architecture cannot control, so "
                         "the cost-of-being-tight outweighs the worker $.")

    ap.add_argument("--out_dir", default="checkpoints/grpo")
    ap.add_argument("--save_every", type=int, default=25,
                    help="Per-step ckpt cadence (matches production).")
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42,
                    help="production seed (deterministic batch order + "
                         "train/test split).")

    # ---- Architecture injection (default = none, the SFT→GRPO method) -----
    ap.add_argument(
        "--inject_mode", choices=["none", "uniform_arch", "family_stratified"],
        default="none",
        help="GRPO architecture injection mode. "
             "`none` (default, production): pure on-policy GRPO — the SFT "
             "prior IS the architectural prior, no injection. (--bench "
             "forces this regardless.) "
             "`family_stratified`: pick K distinct canonical families per "
             "step (legacy --dataset path, from-scratch runs). "
             "`uniform_arch`: uniform-over-arch.",
    )
    ap.add_argument(
        "--inject_k", type=int, default=6,
        help="Number of injected archs per GRPO step. Must be ≤ G (group "
             "size) and ≤ pool size. For family_stratified ≤ 42 (number "
             "of canonical families). Ignored if --inject_k_schedule set.",
    )
    ap.add_argument(
        "--inject_k_schedule", default="",
        help="Comma-separated K curriculum across epochs (e.g. '6,4,2' for "
             "3 epochs with K=6 in ep0, K=4 in ep1, K=2 in ep2). Length "
             "must equal --epochs. Overrides --inject_k when set.",
    )
    ap.add_argument(
        "--max_concurrent_runs", type=int, default=128,
        help="Max parallel arch executions per GRPO step. Production "
             "B×G=16×8=128 traces/step run in a single wave at "
             "concurrency 128 (bottlenecked by API RPS, not the pool). "
             "Qwen probed 100%% clean @128.",
    )
    # Tail-latency caps (production budget — arch-attributable, kept tight
    # so cost is a reward signal; library ArchSpec stays loose at 20/20
    # for eval/tests).
    ap.add_argument("--safety_max_cycles", type=int, default=8,
                    help="Max communication cycles (production: 8). "
                         "Arch-attributable cap — hitting it lowers reward, "
                         "not masked.")
    ap.add_argument("--safety_max_steps", type=int, default=16,
                    help="Max ReAct steps per agent turn (production: 16).")
    ap.add_argument("--wall_clock_timeout_s", type=float, default=900.0,
                    help="Hard per-trace wall-clock cap (seconds). "
                         "Re-derived after --max_new_tokens bumped to 8192: "
                         "average trace = ~12 worker calls × ~24s/call = "
                         "~290s; long-tail traces with several saturating "
                         "calls can hit ~600-800s. 900s sets a 3x headroom "
                         "over typical, kills only the truly pathological "
                         "tail. Total step wall ≈ 900s + GRPO update ≈ 16min "
                         "worst case under max_concurrent=128 single wave.")
    ap.add_argument("--max_llm_calls_per_trace", type=int, default=32,
                    help="Soft cap on total LLM calls (agent.run + Synth) "
                         "inside one trace (production: 32). 0 disables. "
                         "When hit, the trace "
                         "terminates with hit_call_cap=True; this is an "
                         "architecture-attributable cap (n_arch_caps_hit "
                         "increments) — advantage flows normally from the "
                         "heuristic_extract result rather than being masked. "
                         "Useful to prevent tool-heavy outliers from "
                         "dominating step wall time.")
    ap.add_argument("--history_cycles", type=int, default=0,
                    help="Cap how many past cycles of transcript each agent's "
                         "incoming prompt slice contains. 0 = unlimited "
                         "(default). K>0 keeps only cycles "
                         ">= current_cycle - K + 1. Synth always sees the "
                         "full trace regardless of this flag.")

    # ---- (task, arch) result cache ---------------------------------------
    ap.add_argument(
        "--cache_path", default=None,
        help="JSONL file for the per-run (task, arch) → reward cache. "
             "Defaults to `<out_dir>/arch_cache.jsonl`. Set to empty "
             "string '' to DISABLE the cache (every sample re-runs).",
    )
    ap.add_argument(
        "--cache_reuse_prob", type=float, default=1.0,
        help="Probability of returning a cached result on a hit. "
             "1.0 (default) = always reuse. 0.0 = effectively disabled "
             "(every sample treated as miss, cache still writes but never "
             "returns hits — useful for ablations). Intermediate values "
             "trade API spend for stochastic exploration: 0.8 = save ~80%%"
             " of duplicate work, still get fresh samples on 20%% of hits.",
    )

    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default="arch_policy")
    ap.add_argument("--wandb_run", default=None)
    ap.add_argument(
        "--strict_tools", action=argparse.BooleanOptionalAction, default=True,
        help="Fail loudly at startup if SERPER_API_KEY is missing (default "
             "ON). Pass --no-strict_tools to acknowledge that web_search / "
             "arxiv_search (and the Serper fallback path of "
             "wikipedia_search) will return offline-stub strings — a "
             "stub-degraded run still prints a one-time stderr WARN and "
             "every trace's `search_stub_counts` telemetry is non-zero.",
    )

    args = ap.parse_args()

    import random as _random
    from arch_policy import (
        ARCH, ArchitectureHead, GRPOBatch, MultiAgentExecutor,
        TRAIN, bench, load_head_checkpoint, load_tokenizer, seed_all,
        train_grpo,
    )
    from arch_policy.data.tasks import load_huggingface, load_local_synthetic
    from arch_policy.executor.tools import preflight_tools

    # Preflight: assert search tools have a working key BEFORE we burn an
    # hour of LLM calls on stub-degraded traces. --no-strict_tools to skip
    # (still WARNs at first stub return + counts in trace telemetry).
    if args.strict_tools:
        preflight_tools()
    else:
        print("[grpo] --no-strict_tools: search tools may degrade silently; "
              "watch eng/search_stub_total in step logs.", flush=True)

    seed_all(args.seed)

    # --- Load tasks ---
    if args.bench is not None:
        # Per-bench plugin path: loader + grader come from `bench.get(name)`.
        # Inject is forced off — SFT pool IS the architectural prior.
        adapter = bench.get(args.bench)
        tasks = adapter.load_split(args.split, train_ratio=args.train_ratio,
                                   seed=args.seed)
        if args.n and args.n < len(tasks):
            _random.Random(args.seed).shuffle(tasks)
            tasks = tasks[: args.n]
        print(f"[grpo] bench={args.bench} loaded {len(tasks)} "
              f"{args.split} tasks across {len(adapter.subdomains)} subdomains")
        if args.inject_mode != "none":
            print(f"[grpo] --bench forces --inject_mode none "
                  f"(was {args.inject_mode!r})")
            args.inject_mode = "none"
    elif (args.dataset or "gsm8k") == "synthetic":
        tasks = load_local_synthetic(n_per_family=max(1, args.n // 3), seed=args.seed)[: args.n]
    elif args.dataset == "mixed_stress":
        # Composition:
        #   Agent 60%: BrowseComp 30% + HLE 30%        (unsaturated)
        #   STEM  30%: GPQA-Diamond 15% + PHYBench 10% + MATH-500-L5 5%
        #   Code  10%: LiveCodeBench Hard 10%
        # Target SOTA ≈ 40-50% across sources (sweet spot for RL signal).
        # Scales with --n; n=1000 gives the exact percentages above.
        n = args.n
        composition = {
            "browsecomp":     int(n * 0.30),   # web agent core
            "hle":            int(n * 0.30),   # frontier reasoning
            "gpqa":           int(n * 0.15),   # graduate STEM (~50% SOTA)
            "phybench":       int(n * 0.10),   # physics olympiad
            "math":           int(n * 0.05),   # MATH-500 L5 (via level_filter)
            "livecodebench":  int(n * 0.10),   # competitive coding (hard)
        }
        # Per-source kwargs:
        #   - gpqa: loader hardcodes aradhye/gpqa_diamond (198 rows).
        #   - math: level_filter=(5,) restricts to MATH-500 hardest tier.
        #   - livecodebench: loader uses JameSand/livecodebench-v6 mirror
        #     (131 rows, no difficulty field to filter on; all are hard).
        per_source_kw = {
            "math": {"level_filter": (5,)},
        }
        tasks = []
        per_source_loaded = {}
        for src, k in composition.items():
            if k <= 0:
                continue
            try:
                s = load_huggingface(
                    src, split=args.split, n=k, seed=args.seed,
                    **per_source_kw.get(src, {}),
                )
                tasks.extend(s)
                per_source_loaded[src] = len(s)
                print(f"[grpo] mixed_stress: loaded {len(s)}/{k} from {src}")
            except Exception as e:
                print(f"[grpo] mixed_stress: FAILED to load {src}: {type(e).__name__}: {e}")
                per_source_loaded[src] = 0
        if not tasks:
            raise RuntimeError(
                "mixed_stress: all sources failed to load. Check network "
                "and dataset access (HF_TOKEN for hle, etc)."
            )
        # Trim to exactly args.n (in case loaders returned fewer total)
        rng_mix = _random.Random(args.seed)
        rng_mix.shuffle(tasks)
        tasks = tasks[:args.n]
        print(f"[grpo] mixed_stress total: {len(tasks)} (target {args.n}) "
              f"per_source={per_source_loaded}")
    elif args.bench is None:
        dataset = args.dataset or "gsm8k"
        tasks = load_huggingface(dataset, split=args.split, n=args.n, seed=args.seed)
        print(f"[grpo] loaded {len(tasks)} tasks ({dataset}/{args.split})")

    # --- ArchSpec overrides (tail-latency caps + optional model pool) ---
    from dataclasses import replace as _replace
    model_names = (tuple(m.strip() for m in args.worker_models.split(",") if m.strip())
                   if args.worker_models else ARCH.model_names)
    arch_spec = _replace(
        ARCH,
        safety_max_cycles=args.safety_max_cycles,
        safety_max_steps=args.safety_max_steps,
        model_names=model_names,
    )
    print(f"[grpo] ArchSpec: cycles={arch_spec.safety_max_cycles}, "
          f"steps={arch_spec.safety_max_steps}, n_models={arch_spec.n_models}")

    # --- Worker(s) + executor ---
    worker = _build_worker(args)
    worker_pool = None
    if arch_spec.n_models > 1:
        from _common import build_worker_pool
        worker_pool = build_worker_pool(
            arch_spec.model_names,
            timeout=args.worker_timeout,
            temperature=args.worker_temperature,
            thinking=args.worker_thinking,
        )
        # `worker` (single) still serves Synth; agents dispatch via pool.
        worker = worker_pool[arch_spec.model_names[0]]
        print(f"[grpo] model pool ({arch_spec.n_models}): "
              f"{', '.join(arch_spec.model_names)} — per-agent model "
              f"dimension ACTIVE")
    executor = MultiAgentExecutor(worker=worker,
                                  spec=arch_spec,
                                  worker_pool=worker_pool,
                                  max_new_tokens_per_call=args.max_new_tokens,
                                  wall_clock_timeout_s=args.wall_clock_timeout_s,
                                  max_llm_calls_per_trace=args.max_llm_calls_per_trace,
                                  history_cycles=args.history_cycles)
    print(f"[grpo] worker: {args.worker}={args.worker_model} "
          f"(wall_clock_timeout={args.wall_clock_timeout_s}s, "
          f"max_calls_per_trace={args.max_llm_calls_per_trace or 'unlimited'}, "
          f"history_cycles={args.history_cycles or 'unlimited'})")

    # --- Head ---
    tokenizer = load_tokenizer(args.head_model)
    if args.lora_rank > 0:
        mode_str = f"LoRA (rank={args.lora_rank})"
    elif args.freeze_backbone:
        mode_str = "frozen backbone (heads only)"
    else:
        mode_str = "full fine-tune"
    print(f"[grpo] trainability mode: {mode_str}; gradient_checkpointing={args.gradient_checkpointing}")
    model = ArchitectureHead(
        backbone_name=args.head_model,
        arch_spec=arch_spec,
        freeze_backbone=args.freeze_backbone,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_checkpointing=args.gradient_checkpointing,
        torch_dtype=args.head_dtype if args.head_device.startswith("cuda") else "float32",
    )
    if args.head_ckpt is None:
        print("[grpo] mode: from-scratch (no --head_ckpt; typed heads use "
              "random init). Inject curriculum + entropy bonus drive early "
              "exploration. Recommended: --freeze_backbone.")
        if not args.freeze_backbone and args.lora_rank == 0:
            print("[grpo] WARNING: from-scratch + full-FT backbone means "
                  "~4B params train via sparse RL — extremely slow. Strongly "
                  "consider --freeze_backbone (heads-only, ~1M params).")
    else:
        load_head_checkpoint(model, args.head_ckpt)
        print(f"[grpo] mode: SFT warm-start from {args.head_ckpt}")

    # Cross-cutting LR sanity: full-FT 4B params + LLM-default lr=2e-5
    # is too hot for RL (typical full-FT-RL lr 5e-6 ~ 1e-5). LoRA-RL or
    # heads-only-RL @ 2e-5 is fine — those have far fewer trainable
    # params and the lr is well within the LoRA-RL sweet spot. The
    # from-scratch WARN above catches the easy case; this catches the
    # `warm-start + full-FT` corner the original WARN missed.
    if (args.lora_rank == 0 and not args.freeze_backbone
            and args.lr > 5e-6):
        print(f"[grpo] WARNING: full-FT backbone (lora_rank=0, "
              f"freeze_backbone=False) with --lr {args.lr:g} is "
              f"aggressive for RL fine-tuning. Typical full-FT-RL lr "
              f"is 5e-6 ~ 1e-5; expect frequent grad-NaN skips. "
              f"Consider --lora_rank 32 (LoRA) or --lr 5e-6.")
    model.to(args.head_device)
    model.train()
    n_trainable = model.trainable_parameters()
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[grpo] trainable {n_trainable:,} / {n_total:,} "
          f"({100 * n_trainable / max(1, n_total):.2f}%)")

    # --- TrainSpec override ---
    spec = replace(
        TRAIN,
        grpo_group_size=args.G,
        grpo_lr=args.lr,
        grpo_batch_size=args.batch_size,
        grpo_entropy_weight=args.entropy_weight,
        advantage_cost_bonus_scale=args.cost_bonus_scale,
        tokenizer_max_len=args.max_seq_len,
    )

    # --- Wandb ---
    wb = None
    if args.wandb:
        try:
            import wandb
            wb = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run or f"grpo_{args.dataset}_{int(time.time())}",
                config=vars(args),
            )
        except ImportError:
            print("[grpo] wandb not installed; skipping logging")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Build batches ---
    # Group tasks into mini-batches of size args.batch_size; iterate args.epochs
    # times. SHUFFLE the task list at the START OF EACH EPOCH so the head sees
    # different mini-batch compositions (was a real bug: prior version had a
    # "naive shuffle: rotate the list" comment but did NOT actually shuffle,
    # so all epochs used the same task order → over-fit to sequence head).
    rng = _random.Random(args.seed)
    tasks_epoch = list(tasks)
    batches = []
    for epoch in range(args.epochs):
        rng.shuffle(tasks_epoch)
        for i in range(0, len(tasks_epoch), args.batch_size):
            chunk = tasks_epoch[i : i + args.batch_size]
            if not chunk:
                continue
            batches.append(GRPOBatch(
                task_texts=[t.task for t in chunk],
                gold_answers=[t.gold_answer for t in chunk],
                task_samples=list(chunk),
            ))
    print(f"[grpo] total {len(batches)} GRPO steps (G={args.G})")

    # ---- inject mode resolution -------------------------------------------
    inject_pool_arg = None
    inject_family_stratified_arg = False
    if args.inject_mode == "uniform_arch":
        from arch_policy import default_inject_pool
        inject_pool_arg = default_inject_pool()
        mode_label = f"uniform_arch (pool={len(inject_pool_arg)})"
    elif args.inject_mode == "family_stratified":
        inject_family_stratified_arg = True
        from arch_policy.architecture.library import CANONICAL_FAMILIES
        mode_label = f"family_stratified ({len(CANONICAL_FAMILIES)} families)"
    else:
        mode_label = "none (pure on-policy)"

    # ---- per-step K schedule (curriculum across epochs) ------------------
    n_batches_per_epoch = max(1, len(batches) // args.epochs)
    inject_k_per_step = None
    # WARN if user passed a curriculum that's about to be silently
    # discarded (e.g. --bench forces inject_mode=none).
    if args.inject_k_schedule and args.inject_mode == "none":
        print(f"[grpo] WARN: --inject_k_schedule={args.inject_k_schedule!r} "
              f"is being IGNORED because --inject_mode=none. Remove the "
              f"flag or set --inject_mode={{uniform_arch,family_stratified}} "
              f"to actually use it.", flush=True)
    if args.inject_k_schedule and args.inject_mode != "none":
        k_list = [int(x.strip()) for x in args.inject_k_schedule.split(",") if x.strip()]
        if len(k_list) != args.epochs:
            raise ValueError(
                f"--inject_k_schedule has {len(k_list)} values but --epochs={args.epochs}"
            )
        inject_k_per_step = []
        for ep, k in enumerate(k_list):
            n_steps_this_epoch = n_batches_per_epoch
            if ep == args.epochs - 1:
                # last epoch absorbs any remainder
                n_steps_this_epoch = len(batches) - len(inject_k_per_step)
            inject_k_per_step.extend([k] * n_steps_this_epoch)
        # Real raise (not assert) so the invariant survives python -O.
        if len(inject_k_per_step) != len(batches):
            raise RuntimeError(
                f"inject_k_per_step has {len(inject_k_per_step)} entries "
                f"but expected {len(batches)} (batches). Bug in the "
                f"per-epoch absorbed-remainder logic above."
            )
        print(f"[grpo] inject mode = {mode_label} | K curriculum {k_list} over {args.epochs} epochs")
    else:
        print(f"[grpo] inject mode = {mode_label} | K constant = {args.inject_k}")

    # Per-bench reward_fn wraps adapter.grade(judge=...). Legacy
    # compute_reward is used when --bench is unset.
    reward_fn = None
    if args.bench is not None:
        # Only build the judge for benches that grade with one (HLE /
        # FrontierScience). Exec-graded benches (LiveCodeBench) skip it,
        # so the default judge_model never forces a GpuGeek key they
        # don't need.
        judge_worker = _build_judge(args) if adapter.needs_judge() else None
        if adapter.needs_judge() and judge_worker is None:
            print(f"[grpo] WARN: bench={args.bench} wants a judge but "
                  f"--judge_model unset — using rule-grader fallback.")
        reward_fn = bench.make_reward_fn(adapter, judge=judge_worker)
        print(f"[grpo] reward_fn = {args.bench}.grade "
              f"(judge={args.judge}={args.judge_model if adapter.needs_judge() else 'n/a (exec-graded)'})")

    # (task, arch) → reward cache. Per-run scope. Empty --cache_path
    # disables; otherwise default to <out_dir>/arch_cache.jsonl.
    arch_cache = None
    if args.cache_path != "":
        from arch_policy.training.arch_cache import ArchCache
        cp = (args.cache_path if args.cache_path
              else str(out_dir / "arch_cache.jsonl"))
        arch_cache = ArchCache(cp, reuse_prob=args.cache_reuse_prob,
                               seed=args.seed)
        print(f"[grpo] arch_cache: path={cp}  reuse_prob={args.cache_reuse_prob}  "
              f"loaded={len(arch_cache)} entries")
    else:
        print(f"[grpo] arch_cache: DISABLED (--cache_path '')")

    train_grpo(
        model=model,
        tokenizer=tokenizer,
        batches=batches,
        executor=executor,
        spec=spec,
        out_dir=out_dir,
        device=args.head_device,
        log_every=args.log_every,
        save_every=args.save_every,
        wandb_run=wb,
        inject_pool=inject_pool_arg,
        inject_k=args.inject_k if args.inject_mode != "none" else 0,
        inject_k_per_step=inject_k_per_step,
        inject_family_stratified=inject_family_stratified_arg,
        inject_seed=args.seed,
        max_concurrent_runs=args.max_concurrent_runs,
        reward_fn=reward_fn,
        arch_cache=arch_cache,
    )
    if wb is not None:
        wb.finish()
    print(f"[grpo] DONE — checkpoints under {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
