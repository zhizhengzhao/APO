"""Build the per-CATEGORY candidate pool (stage 1 of the 0.5-difficulty flow).

Each category pools several HARD sources so no single benchmark identity
dominates, then: gradeable-filter -> dedup -> cap -> freeze an OVERSIZED
candidate pool to data/categories/_candidates/<cat>.jsonl with provenance.

Stage 2 (`scripts/difficulty_probe.py`) runs the real worker pool over these
candidates, measures per-problem pass_rate under the SFT-prior architecture
distribution, and selects the final 500 centered on pass_rate≈0.5 ->
data/categories/<cat>.jsonl (the frozen corpus the `category` bench reads).

Categories (functional axes: verification x knowledge-intensity):
  code       executable      self-contained  LiveCodeBench(med+hard) + MBPP + HumanEval
  math       symbolic-check  self-contained  AIME + MATH(L4-5) + Omni-MATH(hard) + Olympiad + PHYBench
  knowledge  multiple-choice knowledge-int.  GPQA-Diamond + MMLU-Pro(STEM)
  reasoning  exact/MC        logic-intensive BBH-hard + MuSR + ReClor + ARC-Chal

`legacy` mode reproduces the old direct 500-pick (metadata difficulty buckets);
`candidates` mode (default) builds the oversized pool for the probe.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from arch_policy.data.tasks import load_huggingface, TaskSample
from arch_policy.reward.grade import grade  # for a cheap gradeability sanity check

# Per-category source plan: (loader_name, per_source_cap, loader_kwargs).
# Caps oversample on purpose; stage 2 (difficulty_probe) trims to 500 at ≈0.5.
# Sources are deliberately HARD — easy ones (HumanEval/MBPP/MATH-L<=3) saturated
# the worker at ~0.9 and got dropped by the probe, so they carry small caps
# only for wording diversity.
CATEGORY_SOURCES = {
    "code": [
        # LCB medium+hard is the discriminating core; easy LCB excluded.
        ("livecodebench", 750, {"difficulties": ("medium", "hard")}),
        ("mbpp", 150, {}),
        ("humaneval", 100, {}),
    ],
    "math": [
        # Probe (2026-06-04) showed AIME + MATH-L5 + Omni>=5 SATURATE the
        # ensemble at pass_rate=1.0 (71% of a random sample). Shift to the
        # genuinely ensemble-hard tail: Omni-MATH difficulty>=6 (1289 avail)
        # as the core, + olympiad/physics; keep only a little AIME for
        # diversity (the probe drops whatever still saturates).
        ("omni_math", 520, {"min_difficulty": 6.0}),
        ("olympiad", 280, {}),                       # IMO/IMC-level
        ("phybench", 220, {}),                        # physics olympiad
        ("aime", 160, {}),                           # diversity; probe trims
    ],
    "knowledge": [  # MMLU-Pro STEM pulled per-category below via special handling
        ("gpqa", 198, {}),
    ],
    "reasoning": [
        # v1 (HLE + BrowseComp) probed to mean pass_rate=0.198 — 302/500 tasks
        # saturated at 0.0 (frontier-hard, no architectural signal). Replaced
        # with GRADED-difficulty, exact/MC-gradeable logical-reasoning sources
        # so the probe can actually find a 0.4-0.6 band (no judge needed):
        #   BBH hard subtasks : deduction / tracking / sequence — exact match
        #   MuSR              : multistep soft reasoning (narratives) — MC
        #   ReClor            : LSAT/GMAT logical reasoning — MC
        #   ARC-Challenge     : science reasoning easy tail (probe balances) — MC
        ("bbh", 110, {"subset": "logical_deduction_seven_objects"}),
        ("bbh", 110, {"subset": "logical_deduction_five_objects"}),
        ("bbh", 110, {"subset": "tracking_shuffled_objects_seven_objects"}),
        ("bbh", 110, {"subset": "tracking_shuffled_objects_five_objects"}),
        ("bbh", 110, {"subset": "dyck_languages"}),
        ("bbh", 110, {"subset": "word_sorting"}),
        ("bbh", 110, {"subset": "geometric_shapes"}),
        ("bbh", 110, {"subset": "salient_translation_error_detection"}),
        ("musr", 260, {"subset": "murder_mysteries"}),
        ("musr", 260, {"subset": "object_placements"}),
        ("musr", 260, {"subset": "team_allocation"}),
        ("reclor", 500, {}),
        ("arc", 260, {"subset": "ARC-Challenge"}),
    ],
}

# Per-category cap on the frozen candidate pool (after gradeable+dedup).
# Bigger pool = more probe cost but more room to hit 500 in the 0.5 band.
DEFAULT_N_CAND = 900

# Hard STEM MMLU-Pro categories to back-fill the knowledge pool (harder MC,
# up to 10 options, reasoning-heavy — ~15% below plain MMLU accuracy).
MMLU_PRO_STEM_CATEGORIES = [
    "math", "physics", "chemistry", "biology",
    "computer science", "engineering", "health",
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())[:400]


def _trim_code_tests(t: TaskSample, max_tests: int = 12, max_chars: int = 20000) -> None:
    """LiveCodeBench test cases decode to huge blobs (some problems carry
    100s of MB of test I/O). Cap count + per-test size so the frozen corpus
    stays small; grading is all-or-nothing + short-circuits, so a dozen
    tests is still a strong correctness signal."""
    if t.family != "livecodebench":
        return
    raw = t.metadata.get("tests")
    if not raw:
        return
    try:
        cases = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, ValueError):
        return
    if not isinstance(cases, list):
        return
    kept = []
    for c in cases[:max_tests]:
        if isinstance(c, dict):
            inp, outp = str(c.get("input", "")), str(c.get("output", ""))
            if len(inp) <= max_chars and len(outp) <= max_chars:
                kept.append(c)
    t.metadata["tests"] = json.dumps(kept)


def _difficulty(t: TaskSample) -> str:
    """Best-effort difficulty bucket from metadata; 'unknown' if absent."""
    md = t.metadata or {}
    for k in ("difficulty", "level", "Level"):
        if k in md and md[k] not in (None, ""):
            v = str(md[k]).lower()
            if any(x in v for x in ("easy", "1", "2")):
                return "easy"
            if any(x in v for x in ("medium", "med", "3")):
                return "medium"
            if any(x in v for x in ("hard", "difficult", "4", "5", "olympiad")):
                return "hard"
            return v
    return "unknown"


def load_category(cat: str, seed: int) -> list[TaskSample]:
    pool: list[TaskSample] = []
    for src, cap, kw in CATEGORY_SOURCES[cat]:
        try:
            kw = dict(kw)
            if src == "livecodebench":
                # Use the bench adapter (official 1055, carries difficulty),
                # not the small JameSand mirror that load_huggingface points at.
                from arch_policy import bench as _bench
                difficulties = kw.pop("difficulties", None)
                ts = _bench.get("livecodebench").load_split(
                    "train", train_ratio=1.0, seed=seed)
                if difficulties:
                    allow = {d.lower() for d in difficulties}
                    ts = [t for t in ts
                          if str(t.metadata.get("subject", "")).lower() in allow]
                ts = ts[:cap]
            else:
                ts = load_huggingface(src, split="train", n=cap, seed=seed, **kw)
            for t in ts:
                t.metadata = dict(t.metadata or {}); t.metadata["source"] = src
            pool += ts
            print(f"  [{cat}] {src}: +{len(ts)}")
        except Exception as e:  # noqa: BLE001
            print(f"  [{cat}] {src}: FAILED {type(e).__name__}: {e}")
    if cat == "knowledge":
        # Back-fill with MMLU-Pro STEM categories (harder MC than plain MMLU).
        per_cat = 110  # ~7 categories x 110 ≈ 770 + 198 GPQA → ~960 candidates
        for c in MMLU_PRO_STEM_CATEGORIES:
            try:
                ts = load_huggingface("mmlu_pro", split="test", n=per_cat,
                                      seed=seed, subset=c)
                for t in ts:
                    t.metadata = dict(t.metadata or {})
                    t.metadata["source"] = f"mmlu_pro:{c}"
                pool += ts
            except Exception as e:  # noqa: BLE001
                print(f"  [knowledge] mmlu_pro:{c} FAILED {type(e).__name__}: {e}")
        print(f"  [knowledge] total pooled = {len(pool)}")
    return pool


def gradeable(t: TaskSample) -> bool:
    """Keep only tasks our graders can score (must have gold/test material)."""
    if not (t.task or "").strip():
        return False
    if t.family in ("humaneval", "mbpp"):
        return bool(t.metadata.get("test") or t.metadata.get("prompt"))
    if t.family == "livecodebench":
        return bool(t.metadata.get("tests"))
    return bool((t.gold_answer or "").strip())


def _collect_unique(cat: str, seed: int) -> list[TaskSample]:
    """Collect → gradeable-filter → dedup-by-normalized-text for one category."""
    pool = [t for t in load_category(cat, seed) if gradeable(t)]
    seen, uniq = set(), []
    for t in pool:
        k = _norm(t.task)
        if k and k not in seen:
            seen.add(k)
            uniq.append(t)
    return uniq


def _build_candidates(cats, args, rng) -> dict:
    """Stage 1: write an oversized, gradeable, deduped candidate pool per cat."""
    out = Path(args.out_dir) / "_candidates"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for cat in cats:
        print(f"\n=== candidates [{cat}] ===")
        uniq = _collect_unique(cat, args.seed)
        print(f"  gradeable+dedup: {len(uniq)} unique")
        rng.shuffle(uniq)
        picked = uniq[: args.n_cand]
        for t in picked:
            _trim_code_tests(t)
        recs = []
        for i, t in enumerate(picked):
            recs.append({
                "cand_id": f"{cat}_{i}",
                "task": t.task, "gold_answer": t.gold_answer, "family": t.family,
                "task_id": t.task_id, "metadata": t.metadata,
                "category": cat, "difficulty": _difficulty(t),
            })
        fpath = out / f"{cat}.jsonl"
        with open(fpath, "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[cat] = {
            "n_candidates": len(recs),
            "sources": dict(Counter(r["metadata"].get("source", r["family"]) for r in recs)),
            "difficulty": dict(Counter(r["difficulty"] for r in recs)),
        }
        print(f"  -> wrote {fpath} ({len(recs)} candidates)")
    return summary


def _build_legacy(cats, args, rng) -> dict:
    """Old path: metadata-difficulty-stratified direct pick of N → frozen corpus."""
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for cat in cats:
        print(f"\n=== collecting [{cat}] (legacy) ===")
        uniq = _collect_unique(cat, args.seed)
        print(f"  gradeable+dedup: {len(uniq)} unique")
        by_diff = defaultdict(list)
        for t in uniq:
            by_diff[_difficulty(t)].append(t)
        for v in by_diff.values():
            rng.shuffle(v)
        picked, buckets = [], list(by_diff.values())
        bi = 0
        while len(picked) < args.n and any(buckets):
            b = buckets[bi % len(buckets)]
            if b:
                picked.append(b.pop())
            bi += 1
            if all(not b for b in buckets):
                break
        picked = picked[: args.n]
        for t in picked:
            _trim_code_tests(t)
        rng.shuffle(picked)
        n_train = int(len(picked) * args.train_ratio)
        recs = []
        for i, t in enumerate(picked):
            recs.append({
                "task": t.task, "gold_answer": t.gold_answer, "family": t.family,
                "task_id": t.task_id, "metadata": t.metadata,
                "category": cat, "difficulty": _difficulty(t),
                "split": "train" if i < n_train else "test",
            })
        fpath = out / f"{cat}.jsonl"
        with open(fpath, "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[cat] = {
            "n": len(recs),
            "train": sum(r["split"] == "train" for r in recs),
            "test": sum(r["split"] == "test" for r in recs),
            "sources": dict(Counter(r["metadata"].get("source", r["family"]) for r in recs)),
            "difficulty": dict(Counter(r["difficulty"] for r in recs)),
        }
        print(f"  -> wrote {fpath} ({len(recs)} problems)")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("candidates", "legacy"), default="candidates",
                    help="candidates: oversized pool for the difficulty probe "
                         "(stage 1). legacy: direct metadata-difficulty 500-pick.")
    ap.add_argument("--n", type=int, default=500, help="problems/cat (legacy)")
    ap.add_argument("--n_cand", type=int, default=DEFAULT_N_CAND,
                    help="candidate-pool cap per category (candidates mode)")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="data/categories")
    ap.add_argument("--only", default=None, help="comma list to restrict categories")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    cats = args.only.split(",") if args.only else list(CATEGORY_SOURCES)

    if args.mode == "candidates":
        summary = _build_candidates(cats, args, rng)
    else:
        summary = _build_legacy(cats, args, rng)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
