"""Run all CPU-only smoke tests in sequence.

    python scripts/02_smoke_test.py

This does NOT require GPU or any model download — it exercises:
  - architecture spec / sampler / library / encoder / typed log_probs
  - executor with MockWorker (each baseline + ReAct + Synth + heuristic)
  - SFT typed loss math + tiny-network 40-step descent

If this passes, the v3 architecture-policy code is internally consistent.
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback


MODULES = [
    "tests.test_architecture",
    "tests.test_executor",
    "tests.test_sft_step",
]


def main() -> int:
    failed_total = 0
    for modname in MODULES:
        print(f"\n=== {modname} ===")
        try:
            mod = importlib.import_module(modname)
        except ImportError as e:
            print(f"  cannot import {modname}: {e}")
            failed_total += 1
            continue
        fns = [getattr(mod, k) for k in dir(mod) if k.startswith("test_") and callable(getattr(mod, k))]
        if not fns:
            print("  (no test_ functions)")
            continue
        for fn in fns:
            try:
                fn()
                print(f"  PASS {fn.__name__}")
            except Exception:
                failed_total += 1
                print(f"  FAIL {fn.__name__}")
                traceback.print_exc()
    if failed_total:
        print(f"\n{failed_total} test(s) failed.")
        return 1
    print("\nAll smoke tests pass.")
    return 0


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)        # arch_policy/
    sys.path.insert(0, project_root)            # for `tests.*` modules
    sys.path.insert(0, os.path.join(project_root, "src"))
    sys.exit(main())
