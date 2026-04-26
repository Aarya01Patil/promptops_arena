"""
Phase 6: Evaluate the GRPO-trained agent on the test split.

Loads:
  - base agent: Qwen/Qwen2.5-1.5B-Instruct (frozen weights)
  - LoRA adapter: from --adapter (local dir or HF model repo id)

For each test task:
  1. build agent input (same as training)
  2. agent generates a candidate system prompt
  3. env runs LLM-under-test with that prompt; verify; reward
  4. up to --max-turns retries with the previous attempt visible

Outputs results/trained_agent.json in the same shape as run_baseline.py.
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
from src.envs.promptops_arena.tasks import load_tasks
from src.envs.promptops_arena import llm_under_test
from scripts.train_grpo import build_agent_input  # reuse the exact prompt template


def _load_agent(base_model: str, adapter: str | None):
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, device_map=device,
    )
    if adapter:
        from peft import PeftModel  # type: ignore
        mdl = PeftModel.from_pretrained(mdl, adapter)
    mdl.eval()

    def gen(text: str, max_new_tokens: int = 300) -> str:
        msgs = [
            {"role": "system", "content": "You are a helpful prompt engineer."},
            {"role": "user", "content": text},
        ]
        encoded = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt",
        )
        if hasattr(encoded, "input_ids"):
            ids = encoded.input_ids
        elif isinstance(encoded, dict):
            ids = encoded["input_ids"]
        else:
            ids = encoded
        ids = ids.to(device)
        with torch.no_grad():
            out = mdl.generate(
                input_ids=ids, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    return gen


def _build_followup_input(task: dict, history: List[dict]) -> str:
    """Like build_agent_input but with prior attempts visible (refinement turn)."""
    base = build_agent_input(task)
    if not history:
        return base
    extra = ["", "PRIOR ATTEMPTS (yours, with what the small model produced):"]
    for i, h in enumerate(history, 1):
        extra.append(f"--- attempt {i} (reward={h['reward']:.2f}, correct={h['correct']}) ---")
        extra.append(f"YOUR PROMPT: {h['prompt'][:400]}")
        extra.append(f"MODEL OUTPUT: {h['completion'][:200]}")
    extra.append("")
    extra.append("Improve the system prompt. Output ONLY the new system prompt, no preamble.")
    return base + "\n" + "\n".join(extra)


def evaluate_trained(env, task, agent_gen, max_turns: int = 3) -> Dict[str, Any]:
    history: List[dict] = []
    best_reward = -1.0
    best_components: Dict[str, float] = {}
    correct = False
    edit_turns = 0

    for turn in range(max_turns):
        edit_turns = turn + 1
        ai = build_agent_input(task) if turn == 0 else _build_followup_input(task, history)
        sp = agent_gen(ai).strip() or "Solve this:"
        res = env.execute_prompt(task, sp)
        components = res["reward"]
        total = components["total"]
        is_correct = components["correctness"] >= 1.0

        history.append({
            "prompt": sp,
            "completion": res["completion"],
            "reward": total,
            "correct": is_correct,
        })

        if total > best_reward:
            best_reward = total
            best_components = components

        if is_correct:
            correct = True
            break

    return {
        "task_id": task["id"],
        "task_type": task["type"],
        "policy": "trained_agent",
        "edit_turns": edit_turns,
        "final_reward": best_reward,
        "correct": correct,
        "format_ok": best_components.get("format", 0.0) >= 1.0,
        "components": best_components,
        "trace": history,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default=None,
                   help="Local dir or HF repo id of the LoRA adapter.")
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="results/trained_agent.json")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--per-type", type=int, default=None)
    p.add_argument("--max-turns", type=int, default=3)
    args = p.parse_args()

    os.environ.setdefault("PROMPTOPS_LLM_BACKEND", "transformers")

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

    print(f"[eval_trained] adapter={args.adapter} base={args.base} "
          f"split={args.split} n_tasks={len(tasks)} "
          f"llm_backend={llm_under_test.backend_name()}")

    env = PromptOpsArenaEnvironment(split=args.split, seed=0)
    agent_gen = _load_agent(args.base, args.adapter)

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, task in enumerate(tasks):
        row = evaluate_trained(env, task, agent_gen, max_turns=args.max_turns)
        rows.append(row)
        n_correct = sum(1 for r in rows if r["correct"])
        print(f"  [{i+1}/{len(tasks)}] {task['type']:5s} "
              f"correct={n_correct}/{i+1} "
              f"r={row['final_reward']:+.3f} elapsed={time.time()-t0:.1f}s")

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
        "policy": "trained_agent",
        "adapter": args.adapter,
        "base_model": args.base,
        "split": args.split,
        "llm_backend": llm_under_test.backend_name(),
        "by_type": by_type,
        "overall": overall,
        "rows": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[eval_trained] wrote {out_path}")
    print(f"  overall: {overall['correct']}/{overall['n']} correct, "
          f"mean_reward={overall['mean_reward']:.3f}")
    for tt, d in by_type.items():
        print(f"  {tt:5s}: {d['correct']}/{d['n']} correct, format {d['format']}/{d['n']}")


if __name__ == "__main__":
    main()
