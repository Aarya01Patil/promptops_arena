#!/usr/bin/env bash
# HF Jobs entrypoint: untrained-1.5B-agent baseline on real LLM-under-test.
# Writes results/baseline_untrained_real.json and uploads to model repo.

set -euo pipefail

HF_USERNAME="${HF_USERNAME:-Dar3devil}"
PER_TYPE="${PER_TYPE:-4}"
RESULTS_REPO="${HF_USERNAME}/promptops-arena-agent"

echo "[untrained-eval] HF_USERNAME=${HF_USERNAME} PER_TYPE=${PER_TYPE}"
mkdir -p /workspace
cp -r /code/. /workspace/
cd /workspace

echo "[untrained-eval] python: $(python --version)"
nvidia-smi || echo "no nvidia-smi"

echo "[untrained-eval] installing deps"
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir \
    "transformers==4.55.4" \
    "peft==0.15.2" \
    "accelerate==1.7.0" \
    "datasets==3.6.0" \
    "huggingface_hub>=0.25.0" \
    "jsonschema>=4.20.0" \
    "openenv-core>=0.1.0" \
    "fastapi>=0.110.0" \
    "uvicorn>=0.27.0" \
    "pydantic>=2.0.0"

export PROMPTOPS_LLM_BACKEND=transformers
export PYTHONUTF8=1
export TOKENIZERS_PARALLELISM=false

mkdir -p outputs results

echo "[untrained-eval] running untrained baseline (real LLM, n=$((PER_TYPE * 3)))"
python scripts/run_baseline.py --policy untrained --per-type "${PER_TYPE}" \
    --out results/baseline_untrained_real.json

echo "[untrained-eval] uploading to ${RESULTS_REPO}"
python - <<PY
import os
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="results/baseline_untrained_real.json",
    path_in_repo="baseline_untrained_real.json",
    repo_id="${RESULTS_REPO}",
    repo_type="model",
    commit_message="eval: untrained 1.5B agent baseline (real LLM)",
)
print("[untrained-eval] uploaded")
PY

echo "[untrained-eval] all done."
