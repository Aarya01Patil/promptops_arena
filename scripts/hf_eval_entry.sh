#!/usr/bin/env bash
# HF Jobs entrypoint: evaluate the trained agent + rerun real-LLM baselines on a
# wider per-type sample, then upload all results to the model repo.

set -euo pipefail

HF_USERNAME="${HF_USERNAME:-Dar3devil}"
PER_TYPE="${PER_TYPE:-4}"
ADAPTER_REPO="${ADAPTER_REPO:-${HF_USERNAME}/promptops-arena-agent}"
RESULTS_REPO="${ADAPTER_REPO}"

echo "[eval] HF_USERNAME=${HF_USERNAME} PER_TYPE=${PER_TYPE} ADAPTER_REPO=${ADAPTER_REPO}"
mkdir -p /workspace
cp -r /code/. /workspace/
cd /workspace

echo "[eval] python: $(python --version)"
nvidia-smi || echo "no nvidia-smi"

echo "[eval] installing deps"
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

echo "[eval] downloading adapter ${ADAPTER_REPO}"
python - <<PY
from huggingface_hub import snapshot_download
import os
p = snapshot_download(repo_id="${ADAPTER_REPO}", repo_type="model",
                      local_dir="outputs/grpo-lora",
                      allow_patterns=["adapter_*", "*.json", "*.jinja", "*.txt", "training_log.jsonl"])
print("[eval] adapter at", p)
PY

echo "[eval] running zero-shot baseline (real LLM)"
python scripts/run_baseline.py --policy zero_shot --per-type "${PER_TYPE}" \
    --out results/baseline_zero_shot_real.json

echo "[eval] running CoT baseline (real LLM)"
python scripts/run_baseline.py --policy cot --per-type "${PER_TYPE}" \
    --out results/baseline_cot_real.json

echo "[eval] running trained-agent eval"
python scripts/eval_trained.py --adapter outputs/grpo-lora --per-type "${PER_TYPE}" \
    --out results/trained_agent.json --max-turns 2

echo "[eval] uploading results to ${RESULTS_REPO}"
python - <<PY
import os
from huggingface_hub import HfApi
api = HfApi()
repo = "${RESULTS_REPO}"
for f in [
    "results/baseline_zero_shot_real.json",
    "results/baseline_cot_real.json",
    "results/trained_agent.json",
]:
    api.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f),
                    repo_id=repo, repo_type="model",
                    commit_message=f"eval: {os.path.basename(f)}")
print("[eval] uploaded")
PY

echo "[eval] all done."
