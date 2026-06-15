#!/usr/bin/env bash
# setup.sh — create a clean conda env and install dependencies.
#
# Usage:
#   bash scripts/setup.sh
#
# We use Python 3.11 because some HF deps don't yet support 3.13 well.
# Requires conda (miniconda or anaconda) on PATH.

set -e

ENV_NAME=${ENV_NAME:-arch_policy}

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found in PATH. Source your conda init first."
    exit 1
fi

# 1. Create env if missing
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[setup] env '${ENV_NAME}' already exists. Skipping creation."
else
    echo "[setup] creating conda env '${ENV_NAME}' (python=3.11)..."
    conda create -n "${ENV_NAME}" python=3.11 -y
fi

# 2. Install deps inside the env via `conda run`
echo "[setup] installing pip deps inside ${ENV_NAME}..."
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install -r requirements.txt

# 3. Editable-install the package itself
echo "[setup] installing arch_policy package (editable)..."
conda run -n "${ENV_NAME}" python -m pip install -e .

echo "[setup] writing HF mirror config to env activate hook (set HF_ENDPOINT=https://hf-mirror.com)"
ACTIVATE_DIR="$(conda info --envs | awk -v env="${ENV_NAME}" '$1 == env {print $NF}')/etc/conda/activate.d"
mkdir -p "${ACTIVATE_DIR}"
cat > "${ACTIVATE_DIR}/hf_endpoint.sh" <<'HFEOF'
# Use the HF mirror by default (works inside China). Override by exporting
# HF_ENDPOINT to something else before activating.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HFEOF

echo "[setup] DONE. Activate with: conda activate ${ENV_NAME}"
