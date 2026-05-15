"""Evaluate a model (head or baseline) on a benchmark (v3).

Usage examples
--------------

# Baseline: solver-verifier on GSM8K (200 questions), DeepSeek worker
python scripts/05_evaluate.py \
    --mode baseline --baseline solver_verifier \
    --dataset gsm8k --n 200 \
    --worker openai --worker_model deepseek-chat \
    --out_jsonl results/sv_gsm8k.jsonl

# Our trained head on GSM8K (deterministic mode)
python scripts/05_evaluate.py \
    --mode head --head_ckpt checkpoints/sft_v1/head_step100 \
    --dataset gsm8k --n 200 \
    --worker openai --worker_model deepseek-chat \
    --out_jsonl results/apo_gsm8k.jsonl

# Self-consistency over architectures (sample 5 archs, vote)
python scripts/05_evaluate.py --mode head --head_ckpt ... \
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

# Approximate prices (USD / 1K tokens) — keep up to date if you change models.
TOKEN_PRICES = {
    "deepseek-chat": (0.00027, 0.00110),     # DeepSeek V3 cache miss
    "deepseek-reasoner": (0.00055, 0.00219), # DeepSeek R1
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "qwen3-4b": (0.0, 0.0),                   # local
}


def _price(model: str, n_in: int, n_out: int) -> float:
    p_in, p_out = TOKEN_PRICES.get(model, (0.0, 0.0))
    return n_in / 1000 * p_in + n_out / 1000 * p_out


def _build_worker(args):
    if args.worker == "mock":
        from arch_policy import MockWorker
        return MockWorker(fake_answer=args.mock_answer)
    if args.worker == "openai":
        from arch_policy import OpenAIWorker
        return OpenAIWorker(
            model=args.worker_model,
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
            temperature=args.worker_temperature,
            timeout=args.worker_timeout,
        )
    if args.worker == "hf":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from arch_policy import HFWorker
        tok = AutoTokenizer.from_pretrained(args.worker_model)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(
            args.worker_model,
            torch_dtype="bfloat16" if args.worker_dtype == "bfloat16" else "auto",
            device_map="auto",
        )
        return HFWorker(model=mdl, tokenizer=tok)
    raise ValueError(f"unknown worker {args.worker}")


def _build_archs_for_task(args, task_text, head=None, tokenizer=None, device=None):
    """Either return a fixed-baseline arch, or sample one (or many) from head."""
    if args.mode == "baseline":
        from arch_policy.baselines import get_baseline
        return [get_baseline(args.baseline)]

    # mode == "head"
    import torch
    from arch_policy import sample_arch, to_arch_logits

    enc = tokenizer([task_text], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = head(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    logits = to_arch_logits(out, batch_idx=0)
    # Move to CPU for sampling (uses Python lists internally)
    logits_cpu = type(logits)(
        gate_logits=logits.gate_logits.cpu(),
        role_logits=logits.role_logits.cpu(),
        edge_logits=logits.edge_logits.cpu(),
        seq_scores=logits.seq_scores.cpu(),
    )
    n_samples = max(1, args.self_consistency)
    archs = []
    for _ in range(n_samples):
        archs.append(sample_arch(logits_cpu, deterministic=not args.head_sample))
    return archs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "head"], required=True)
    ap.add_argument("--baseline", default="single",
                    help="baseline name (see arch_policy.baselines.BASELINE_REGISTRY)")
    ap.add_argument("--head_ckpt", default=None, help="path to head checkpoint dir")
    ap.add_argument("--head_model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--head_device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    ap.add_argument("--head_sample", action="store_true",
                    help="stochastic sampling instead of deterministic (needed for self_consistency > 1)")
    ap.add_argument("--self_consistency", type=int, default=1,
                    help="how many architectures to sample per task (head mode only)")

    ap.add_argument("--dataset", choices=["synthetic", "gsm8k", "humaneval", "math"], default="gsm8k")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--worker", choices=["mock", "openai", "hf"], default="openai")
    ap.add_argument("--worker_model", default="deepseek-chat")
    ap.add_argument("--worker_dtype", default="bfloat16")
    ap.add_argument("--worker_temperature", type=float, default=0.0)
    ap.add_argument("--worker_timeout", type=float, default=120.0)
    ap.add_argument("--mock_answer", default="42")

    ap.add_argument("--max_new_tokens", type=int, default=2048,
                    help="Max generated tokens per worker LLM call (ArchSpec.safety_max_tokens_per_call)")

    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    from arch_policy import MultiAgentExecutor, seed_all
    from arch_policy.data.tasks import load_huggingface, load_local_synthetic
    from arch_policy.reward import grade

    seed_all(args.seed)

    # --- Load tasks ---
    if args.dataset == "synthetic":
        tasks = load_local_synthetic(n_per_family=max(1, args.n // 3), seed=args.seed)
        tasks = tasks[: args.n]
    else:
        tasks = load_huggingface(args.dataset, split=args.split, n=args.n, seed=args.seed)
    print(f"[eval] loaded {len(tasks)} tasks from {args.dataset}/{args.split}")

    # --- Build worker ---
    worker = _build_worker(args)
    print(f"[eval] worker = {args.worker}, model = {args.worker_model}")

    # --- Build executor ---
    executor = MultiAgentExecutor(
        worker=worker,
        max_new_tokens_per_call=args.max_new_tokens,
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
            freeze_backbone=True,
            torch_dtype="float32" if args.head_device == "cpu" else "bfloat16",
        )
        load_head_checkpoint(head, args.head_ckpt)
        head.to(args.head_device).eval()
        print(f"[eval] loaded head from {args.head_ckpt}")

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
        score = grade(final_answer, sample)
        n_correct += score

        rec = {
            "task_id": sample.task_id,
            "family": sample.family,
            "gold": sample.gold_answer,
            "prediction": final_answer,
            "score": score,
            "n_archs_voted": len(archs),
            "tokens_in": sum(t.total_input_tokens for t in per_arch_outputs),
            "tokens_out": sum(t.total_output_tokens for t in per_arch_outputs),
            "llm_calls": sum(t.n_llm_calls for t in per_arch_outputs),
            "n_cycles": [t.n_cycles_run for t in per_arch_outputs],
            "n_synth_calls": [t.n_synth_calls for t in per_arch_outputs],
            "via_synth": [t.final_via_synth for t in per_arch_outputs],
            "arch_summary": [t.arch.to_summary() for t in per_arch_outputs],
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
