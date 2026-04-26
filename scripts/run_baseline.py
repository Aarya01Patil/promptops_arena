"""
Phase 4 baselines on the held-out test split.

Three policies:
  zero_shot : "Solve this:" wrapper, no CoT
  cot       : "Think step by step. Final answer in <answer> tags." style
  untrained : Qwen2.5-1.5B-Instruct (no LoRA) writes the system prompt,
              then LLM-under-test runs it. 3 edit turns.

For local CPU dev we run zero_shot/cot only with the stub backend; untrained
is meant for GPU runs (CUDA or HF Jobs).

Usage:
  python scripts\run_baseline.py --policy zero_shot --out results/baseline_zero_shot.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.envs.promptops_arena.server.environment import PromptOpsArenaEnvironment
from src.envs.promptops_arena.models import PromptOpsAction
from src.envs.promptops_arena.tasks import load_tasks
from src.envs.promptops_arena import llm_under_test


ZERO_SHOT_PROMPT = "Solve this:"

COT_PROMPT_BY_TYPE = {
    "math": (
        "Think step by step. After reasoning, put ONLY the final numeric answer "
        "inside <answer>...</answer> tags. Do not include units or words inside the tags."
    ),
    "code": (
        "Write the requested Python function. Reason briefly, then output exactly one "
        "```python ...``` code block containing only the function definition. "
        "Do not include explanations after the code block."
    ),
    "json": (
        "Extract the requested fields. Output exactly one ```json ...``` code block "
        "containing a JSON object that matches the schema. Use the correct types. "
        "No prose."
    ),
}


def _evaluate_zero_shot(env: PromptOpsArenaEnvironment, task: dict) -> Dict[str, Any]:
    res = env.execute_prompt(task, ZERO_SHOT_PROMPT)
    return {
        "task_id": task["id"],
        "task_type": task["type"],
        "policy": "zero_shot",
        "edit_turns": 1,
        "final_reward": res["reward"]["total"],
        "correct": res["reward"]["correctness"] >= 1.0,
        "format_ok": res["reward"]["format"] >= 1.0,
        "components": res["reward"],
    }


def _evaluate_cot(env: PromptOpsArenaEnvironment, task: dict) -> Dict[str, Any]:
    sp = COT_PROMPT_BY_TYPE.get(task["type"], ZERO_SHOT_PROMPT)
    res = env.execute_prompt(task, sp)
    return {
        "task_id": task["id"],
        "task_type": task["type"],
        "policy": "cot",
        "edit_turns": 1,
        "final_reward": res["reward"]["total"],
        "correct": res["reward"]["correctness"] >= 1.0,
        "format_ok": res["reward"]["format"] >= 1.0,
        "components": res["reward"],
    }


def _build_agent_input(task: dict, history: List[dict]) -> str:
    """Build the prompt the agent sees when asked to write a system prompt."""
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
        parts.append("REQUIRED FORMAT: the answer must be a number inside <answer>...</answer> tags.")
    elif task["type"] == "code":
        parts.append("REQUIRED FORMAT: a single ```python ...``` code block defining the requested function.")
    elif task["type"] == "json":
        parts.append("REQUIRED FORMAT: a single ```json ...``` code block with a valid JSON object matching the schema.")
        if "schema" in task:
            parts.append(f"SCHEMA: {json.dumps(task['schema'])}")

    if history:
        parts.append("")
        parts.append("PREVIOUS ATTEMPTS (your earlier prompts and the model's responses):")
        for i, h in enumerate(history, 1):
            parts.append(f"--- attempt {i} (reward={h['reward']:.2f}, correct={h['correct']}) ---")
            parts.append(f"YOUR PROMPT: {h['prompt'][:400]}")
            parts.append(f"MODEL OUTPUT: {h['completion'][:200]}")
        parts.append("")
        parts.append("Improve the system prompt. Output ONLY the new system prompt, no preamble.")
    else:
        parts.append("")
        parts.append("Output ONLY the system prompt, no preamble.")

    return "\n".join(parts)


def _evaluate_untrained_agent(
    env: PromptOpsArenaEnvironment,
    task: dict,
    agent_generate,
    max_turns: int = 3,
) -> Dict[str, Any]:
    history: List[dict] = []
    best_reward = -1.0
    final_components = {}
    correct = False
    edit_turns = 0

    for turn in range(max_turns):
        edit_turns = turn + 1
        agent_input = _build_agent_input(task, history)
        system_prompt = agent_generate(agent_input).strip()
        if not system_prompt:
            system_prompt = ZERO_SHOT_PROMPT

        res = env.execute_prompt(task, system_prompt)
        components = res["reward"]
        total = components["total"]
        is_correct = components["correctness"] >= 1.0

        history.append({
            "prompt": system_prompt,
            "completion": res["completion"],
            "reward": total,
            "correct": is_correct,
        })

        if total > best_reward:
            best_reward = total
            final_components = components

        if is_correct:
            correct = True
            break

    return {
        "task_id": task["id"],
        "task_type": task["type"],
        "policy": "untrained_agent",
        "edit_turns": edit_turns,
        "final_reward": best_reward,
        "correct": correct,
        "format_ok": final_components.get("format", 0.0) >= 1.0,
        "components": final_components,
        "trace": history,
    }


def _make_agent_generate(model_id: str):
    """Returns callable(text) -> generated text. Uses a separate transformers model."""
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)
    mdl.eval()

    def gen(text: str) -> str:
        msgs = [
            {"role": "system", "content": "You are a helpful prompt engineer."},
            {"role": "user", "content": text},
        ]
        encoded = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        if hasattr(encoded, "input_ids"):
            ids = encoded.input_ids
        elif isinstance(encoded, dict):
            ids = encoded["input_ids"]
        else:
            ids = encoded
        ids = ids.to(device)
        with torch.no_grad():
            out = mdl.generate(
                input_ids=ids, max_new_tokens=300, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    return gen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["zero_shot", "cot", "untrained"], required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None, help="cap tasks for quick runs")
    p.add_argument("--per-type", type=int, default=None, help="cap tasks per type (stratified)")
    p.add_argument("--agent-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = p.parse_args()

    tasks = load_tasks(split=args.split)
    if args.per_type:
        bucketed: Dict[str, List[dict]] = {}
        for t in tasks:
            bucketed.setdefault(t["type"], []).append(t)
        sampled: List[dict] = []
        for tt, lst in bucketed.items():
            sampled.extend(lst[: args.per_type])
        tasks = sampled
    if args.limit:
        tasks = tasks[: args.limit]

    print(f"[baseline] policy={args.policy} split={args.split} n_tasks={len(tasks)} "
          f"llm_backend={llm_under_test.backend_name()}")

    env = PromptOpsArenaEnvironment(split=args.split, seed=0)

    agent_gen = None
    if args.policy == "untrained":
        print(f"[baseline] loading agent model: {args.agent_model}")
        agent_gen = _make_agent_generate(args.agent_model)

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, task in enumerate(tasks):
        if args.policy == "zero_shot":
            row = _evaluate_zero_shot(env, task)
        elif args.policy == "cot":
            row = _evaluate_cot(env, task)
        else:
            row = _evaluate_untrained_agent(env, task, agent_gen, max_turns=3)
        rows.append(row)
        if (i + 1) % 5 == 0 or i == len(tasks) - 1:
            n_correct = sum(1 for r in rows if r["correct"])
            print(f"  [{i+1}/{len(tasks)}] correct={n_correct}/{i+1} "
                  f"elapsed={time.time()-t0:.1f}s")

    by_type: Dict[str, Dict[str, int]] = {}
    for r in rows:
        d = by_type.setdefault(r["task_type"], {"n": 0, "correct": 0, "format": 0})
        d["n"] += 1
        d["correct"] += int(r["correct"])
        d["format"] += int(r["format_ok"])

    overall = {
        "n": len(rows),
        "correct": sum(1 for r in rows if r["correct"]),
        "format": sum(1 for r in rows if r["format_ok"]),
        "mean_reward": sum(r["final_reward"] for r in rows) / max(1, len(rows)),
    }

    out = {
        "policy": args.policy,
        "split": args.split,
        "llm_backend": llm_under_test.backend_name(),
        "by_type": by_type,
        "overall": overall,
        "rows": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[baseline] wrote {out_path}")
    print(f"  overall: {overall['correct']}/{overall['n']} correct, mean_reward={overall['mean_reward']:.3f}")
    for tt, d in by_type.items():
        print(f"  {tt:5s}: {d['correct']}/{d['n']} correct, format {d['format']}/{d['n']}")


if __name__ == "__main__":
    main()
