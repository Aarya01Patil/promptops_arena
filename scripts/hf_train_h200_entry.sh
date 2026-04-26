#!/usr/bin/env bash
# HF Jobs entrypoint: bigger GRPO run on H200, then eval new adapter on test split.
# Pushes new adapter + training_log + trained_agent.json to model repo.

set -euo pipefail

HF_USERNAME="${HF_USERNAME:-Dar3devil}"
STEPS="${STEPS:-300}"
BATCH="${BATCH:-8}"
NUM_GENS="${NUM_GENS:-4}"
PER_TYPE="${PER_TYPE:-4}"
MODEL_REPO="${HF_USERNAME}/promptops-arena-agent"

echo "[train-h200] HF_USERNAME=${HF_USERNAME} STEPS=${STEPS} BATCH=${BATCH} NUM_GENS=${NUM_GENS}"
mkdir -p /workspace
cp -r /code/. /workspace/
cd /workspace

echo "[train-h200] python: $(python --version)"
nvidia-smi || echo "no nvidia-smi"

echo "[train-h200] installing deps (trl 0.21 stack)"
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir \
    "trl==0.21.0" \
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

echo "[train-h200] launching GRPO training"
python scripts/train_grpo.py \
    --steps "${STEPS}" \
    --batch "${BATCH}" \
    --num-generations "${NUM_GENS}" \
    --out outputs/grpo-lora \
    --log results/training_log.jsonl

echo "[train-h200] training done. running test-split eval on new adapter."
python scripts/eval_trained.py \
    --adapter outputs/grpo-lora \
    --per-type "${PER_TYPE}" \
    --out results/trained_agent.json \
    --max-turns 2

echo "[train-h200] uploading adapter + log + eval to ${MODEL_REPO}"
python - <<PY
import os
from huggingface_hub import HfApi, create_repo
api = HfApi()
repo_id = "${MODEL_REPO}"
create_repo(repo_id, repo_type="model", exist_ok=True, private=False)

api.upload_folder(
    folder_path="outputs/grpo-lora",
    repo_id=repo_id,
    repo_type="model",
    commit_message="GRPO H200 run: 300 steps, batch=8, G=4",
)
api.upload_file(
    path_or_fileobj="results/training_log.jsonl",
    path_in_repo="training_log.jsonl",
    repo_id=repo_id,
    repo_type="model",
    commit_message="training reward log (h200 run)",
)
api.upload_file(
    path_or_fileobj="results/trained_agent.json",
    path_in_repo="trained_agent.json",
    repo_id=repo_id,
    repo_type="model",
    commit_message="trained-agent eval (h200 adapter)",
)
print(f"[train-h200] uploaded to https://huggingface.co/{repo_id}")
PY

echo "[train-h200] all done."
