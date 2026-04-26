"""
Phase 5: GRPO training of the prompt-engineering agent.

Agent: Qwen/Qwen2.5-1.5B-Instruct + LoRA adapter (trained).
LLM-under-test: Qwen/Qwen2.5-0.5B-Instruct (frozen, env-side).

Each "prompt" the GRPO trainer sees describes a task. The agent's "completion"
is the system prompt it would give to the LLM-under-test. We then run the
LLM-under-test inside the reward function and return the env reward.

Modes:
  --smoke    : tiny config; 2 steps on CPU with stub LLM. Proves plumbing.
  --hf-jobs  : print the `hf jobs run` command for an a10g-large run.
  default    : real GRPO run; expects CUDA + transformers backend.

Outputs:
  outputs/grpo-lora/                        # LoRA adapter
  results/training_log.jsonl                # per-step rewards (custom callback)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Agent-input builder (kept consistent with run_baseline._build_agent_input)
# ---------------------------------------------------------------------------

def build_agent_input(task: dict) -> str:
    parts = [
        "You are a prompt engineer. Your job is to write a SYSTEM PROMPT that, "
        "when given to a small language model along with the task below, will "
        "produce a correct answer in the required format.",
        "",
        f"TASK TYPE: {task['type']}",
        f"TASK: {task['question']}",
        "",
    ]
    if task["type"] == "math":
        parts.append(
            "REQUIRED FORMAT: the final numeric answer must be inside "
            "<answer>...</answer> tags. Just the number, no units."
        )
    elif task["type"] == "code":
        parts.append(
            "REQUIRED FORMAT: a single ```python ...``` code block defining "
            "the requested function. No prose, no examples."
        )
    elif task["type"] == "json":
        parts.append(
            "REQUIRED FORMAT: a single ```json ...``` code block with a valid "
            "JSON object matching the schema."
        )
        if "schema" in task:
            parts.append(f"SCHEMA: {json.dumps(task['schema'])}")

    parts.append("")
    parts.append("Output ONLY the system prompt itself. No preamble, no markdown fences.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Reward wrapper for GRPO
# ---------------------------------------------------------------------------

def make_reward_fn(log_path: Path):
    """
    Returns a callable that GRPOTrainer can use:
        reward_fn(prompts, completions, **kwargs) -> List[float]

    `kwargs` may include the original dataset columns; we use `task` to
    recover the task dict.
    """
    from src.envs.promptops_arena.server.environment import PromptOpsArenaEnvironment
    from src.envs.promptops_arena import llm_under_test  # noqa: F401

    env = PromptOpsArenaEnvironment(split="train", seed=0)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("a", encoding="utf-8")

    def reward_fn(prompts, completions, **kwargs) -> List[float]:
        tasks = kwargs.get("task")
        if tasks is None:
            raise RuntimeError("reward_fn requires 'task' column in dataset")
        if isinstance(tasks, dict):
            tasks = [tasks] * len(completions)

        rewards: List[float] = []
        for completion, task in zip(completions, tasks):
            if isinstance(completion, list):
                # chat-style completion: list of {role, content}
                text = "".join(
                    m.get("content", "") for m in completion
                    if isinstance(m, dict) and m.get("role") == "assistant"
                )
            else:
                text = str(completion)
            res = env.execute_prompt(task, text.strip())
            rewards.append(float(res["reward"]["total"]))
            log_f.write(json.dumps({
                "ts": time.time(),
                "task_id": task.get("id"),
                "task_type": task.get("type"),
                "reward": res["reward"],
                "completion_len": len(text),
            }) + "\n")
            log_f.flush()
        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_dataset():
    from datasets import Dataset
    from src.envs.promptops_arena.tasks import load_tasks

    tasks = load_tasks(split="train")
    rows = [
        {"prompt": build_agent_input(t), "task": t}
        for t in tasks
    ]
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="Tiny CPU run with stub LLM to validate plumbing.")
    p.add_argument("--dry", action="store_true",
                   help="Construct trainer but don't call .train(). Validates API.")
    p.add_argument("--hf-jobs", action="store_true",
                   help="Print HF Jobs launch command and exit.")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--out", default="outputs/grpo-lora")
    p.add_argument("--log", default="results/training_log.jsonl")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--num-generations", type=int, default=4,
                   help="GRPO group size G (completions per prompt).")
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max-prompt-length", type=int, default=512)
    p.add_argument("--max-completion-length", type=int, default=300)
    args = p.parse_args()

    if args.hf_jobs:
        print(_HF_JOBS_HELP)
        return

    if args.smoke:
        os.environ["PROMPTOPS_LLM_BACKEND"] = "stub"

    # Lazy imports so --hf-jobs and --smoke don't require torch/trl up front.
    import torch  # type: ignore
    from transformers import AutoTokenizer  # type: ignore

    try:
        from trl import GRPOConfig, GRPOTrainer  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "trl is required for GRPO training. Install with: pip install trl>=0.21\n"
            f"(import error: {e})"
        )

    use_unsloth = False
    if not args.smoke:
        try:
            from unsloth import FastLanguageModel  # type: ignore  # noqa: F401
            use_unsloth = torch.cuda.is_available()
        except ImportError:
            use_unsloth = False

    print(f"[train_grpo] mode={'smoke' if args.smoke else 'full'} "
          f"cuda={torch.cuda.is_available()} unsloth={use_unsloth}")

    # ---- model ----
    if use_unsloth:
        from unsloth import FastLanguageModel  # type: ignore
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_prompt_length + args.max_completion_length,
            load_in_4bit=True,
            fast_inference=False,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
    else:
        from transformers import AutoModelForCausalLM  # type: ignore
        from peft import LoraConfig, get_peft_model  # type: ignore

        device_map = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, device_map=device_map,
        )
        lora_cfg = LoraConfig(
            r=8 if args.smoke else 16,
            lora_alpha=16 if args.smoke else 32,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj", "v_proj"] if args.smoke else [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- data + reward ----
    dataset = build_dataset()
    if args.smoke:
        dataset = dataset.select(range(min(4, len(dataset))))
    print(f"[train_grpo] dataset rows={len(dataset)}")

    reward_fn = make_reward_fn(Path(args.log))

    # ---- GRPO config ----
    on_gpu = torch.cuda.is_available() and not args.smoke
    per_device_bs = 2 if args.smoke else args.batch
    num_gens = 2 if args.smoke else args.num_generations

    # trl 0.21 GRPOConfig: has max_prompt_length; no generation_batch_size.
    # Build kwargs compatible across 0.21+ (modern fields ignored if unknown).
    cfg_kwargs = dict(
        output_dir=args.out,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=1,
        num_generations=num_gens,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=128 if args.smoke else args.max_completion_length,
        learning_rate=args.lr,
        max_steps=2 if args.smoke else args.steps,
        logging_steps=1,
        save_steps=10_000 if args.smoke else max(1, args.steps // 4),
        bf16=on_gpu,
        fp16=False,
        use_cpu=not on_gpu,
        report_to=[],
        remove_unused_columns=False,
        beta=0.04,
        temperature=1.0,
    )
    # Build, dropping unknown fields if a newer/older trl rejects one.
    import inspect as _inspect
    _allowed = set(_inspect.signature(GRPOConfig.__init__).parameters.keys())
    cfg_kwargs = {k: v for k, v in cfg_kwargs.items() if k in _allowed}
    cfg = GRPOConfig(**cfg_kwargs)

    # trl 0.21 uses `processing_class` for tokenizer-like; older releases used
    # `tokenizer`. Try processing_class first, fall back.
    _tr_params = set(_inspect.signature(GRPOTrainer.__init__).parameters.keys())
    tr_kwargs = dict(
        model=model,
        reward_funcs=[reward_fn],
        args=cfg,
        train_dataset=dataset,
    )
    if "processing_class" in _tr_params:
        tr_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in _tr_params:
        tr_kwargs["tokenizer"] = tokenizer
    trainer = GRPOTrainer(**tr_kwargs)

    if args.dry:
        print("[train_grpo] dry mode: trainer constructed OK; skipping .train()")
        return

    print("[train_grpo] starting training...")
    trainer.train()
    print(f"[train_grpo] saving adapter to {args.out}")
    trainer.save_model(args.out)
    print("[train_grpo] done.")


_HF_JOBS_HELP = """\
# Launch full GRPO training on HF Jobs (a10g-large, ≤2h cap):
hf jobs run --gpu a10g-large --timeout 7200 \\
    --secrets HF_TOKEN \\
    --env PROMPTOPS_LLM_BACKEND=transformers \\
    --workdir /workspace \\
    --upload . \\
    python:3.11 \\
    bash -c "pip install -r requirements.txt && pip install trl peft && \\
             python scripts/train_grpo.py --steps 200 --batch 4 --num-generations 4 \\
             && hf upload <user>/promptops-arena-agent outputs/grpo-lora ."
"""


if __name__ == "__main__":
    main()
