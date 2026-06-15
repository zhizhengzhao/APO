"""Download the head model from HuggingFace into the local cache.

Run once after the env is set up. This avoids surprises during training.

    python scripts/01_download_models.py

Override with HEAD_MODEL env var:

    HEAD_MODEL=Qwen/Qwen3-0.6B python scripts/01_download_models.py

Only the HEAD backbone is downloaded — workers (agents and Synth) use a
remote chat API (GpuGeek by default; see README).
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    # Default to the HF mirror (works inside China). Override with HF_ENDPOINT env.
    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print(f"Using HF endpoint: {os.environ.get('HF_ENDPOINT')}")

    head = os.environ.get("HEAD_MODEL", "Qwen/Qwen3-4B")
    print(f"Downloading head backbone: {head}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface-hub not installed. Run scripts/setup.sh first.")
        return 1

    try:
        path = snapshot_download(
            repo_id=head,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.py",
                "*.safetensors",
                "*.bin",
                "tokenizer.model",
                "tokenizer.json",
                "tokenizer_config.json",
                "merges.txt",
                "vocab.*",
            ],
        )
        print(f"  cached at: {path}")
    except Exception as e:
        print(f"  FAILED: {e}")
        return 2

    print("Head model downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
