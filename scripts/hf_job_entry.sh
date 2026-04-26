#!/usr/bin/env bash
# Entrypoint executed inside the HF Jobs container.
# Expects:
#   /code        -> RO mount of dataset Dar3devil/promptops-arena-src
#   $HF_TOKEN    -> secret, for pushing the trained adapter
#   $HF_USERNAME -> user namespace for the model repo (default: Dar3devil)
#   $STEPS, $BATCH, $NUM_GENS (optional overrides)

set -euo pipefail

HF_USERNAME="${HF_USERNAME:-Dar3devil}"
STEPS="${STEPS:-200}"
BATCH="${BATCH:-4}"
NUM_GENS="${NUM_GENS:-4}"
LOG_LEVEL="${LOG_LEVEL:-info}"
MODEL_REPO="${HF_USERNAME}/promptops-arena-agent"

echo "[entry] HF_USERNAME=${HF_USERNAME} STEPS=${STEPS} BATCH=${BATCH} NUM_GENS=${NUM_GENS}"
echo "[entry] copying source from /code -> /workspace"
mkdir -p /workspace
cp -r /code/. /workspace/
cd /workspace

echo "[entry] python: $(python --version)"
echo "[entry] gpu:"
nvidia-smi || echo "no nvidia-smi"

echo "[entry] installing deps (pinned for trl 0.21 stack)"
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

echo "[entry] launching GRPO training"
python scripts/train_grpo.py \
    --steps "${STEPS}" \
    --batch "${BATCH}" \
    --num-generations "${NUM_GENS}" \
    --out outputs/grpo-lora \
    --log results/training_log.jsonl

echo "[entry] training done. uploading adapter + log to ${MODEL_REPO}"
python - <<'PY'
import os
from huggingface_hub import HfApi, create_repo

api = HfApi()
repo_id = f"{os.environ['HF_USERNAME']}/promptops-arena-agent"
create_repo(repo_id, repo_type="model", exist_ok=True, private=False)

api.upload_folder(
    folder_path="outputs/grpo-lora",
    repo_id=repo_id,
    repo_type="model",
    commit_message="GRPO-trained LoRA adapter",
)
# also upload training log so we can plot reward curves locally
api.upload_file(
    path_or_fileobj="results/training_log.jsonl",
    path_in_repo="training_log.jsonl",
    repo_id=repo_id,
    repo_type="model",
    commit_message="training reward log",
)
print(f"[entry] uploaded to https://huggingface.co/{repo_id}")
PY

echo "[entry] all done."
