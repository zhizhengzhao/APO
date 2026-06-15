"""Build the shared-SFT corpus `data/categories/mixed.jsonl` (cat_mixed bench).

cat_mixed is the union of all 4 category corpora's TRAIN split — the single,
category-agnostic task pool used to train ONE shared SFT prior (the common
warm-start for every category's GRPO run). SFT pairs each task with a random
architecture (no task->arch grounding), so only task wording variety matters
here; the per-category train/test freeze lives in the 4 source files.

Run AFTER `difficulty_probe.py` has frozen the 4 corpora.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

CATEGORIES = ("code", "math", "knowledge", "reasoning")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/categories")
    ap.add_argument("--split", default="train",
                    help="which split to pool into the mixed SFT corpus")
    ap.add_argument("--out", default=None, help="default <data_dir>/mixed.jsonl")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    out = Path(args.out) if args.out else data_dir / "mixed.jsonl"

    recs, per_cat = [], Counter()
    for cat in CATEGORIES:
        path = data_dir / f"{cat}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run difficulty_probe first")
        for ln in open(path):
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("split") != args.split:
                continue
            # mixed is one pool; keep provenance but all rows are SFT 'train'.
            r = {k: r[k] for k in ("task", "gold_answer", "family", "task_id",
                                   "metadata", "category", "difficulty")
                 if k in r}
            r["split"] = "train"
            recs.append(r)
            per_cat[cat] += 1

    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> wrote {out}: {len(recs)} tasks  per_cat={dict(per_cat)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
