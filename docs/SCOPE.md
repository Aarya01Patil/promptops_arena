# PromptOps Arena — Scope (Locked)

> Locked at T+0. Any feature not on this page is OUT OF SCOPE for the 48h hackathon.

## Thesis (one sentence)

An OpenEnv RL environment where an agent learns, via GRPO, to write and iteratively edit prompts that maximize verifiable task success on a frozen LLM-under-test, across math/code/JSON tasks — demonstrating *transferable* prompt-engineering strategy as a learned skill.

## Models (locked)

| Role | Model | Notes |
|---|---|---|
| Agent (trained) | `Qwen/Qwen2.5-1.5B-Instruct` + LoRA | Trained with GRPO via Unsloth, 4-bit |
| LLM-under-test (frozen) | `Qwen/Qwen2.5-0.5B-Instruct` | Never trained. Loaded once at module top. |

## Tasks (locked)

| Type | Source | Count (train) | Count (test, held-out) | Verifier |
|---|---|---|---|---|
| Math | GSM8K subset | 30 | 10 | Exact match on `\boxed{}` or `<answer>` extraction |
| Code | MBPP subset | 20 | 10 | Subprocess `exec` with timeout, run unit tests |
| JSON extraction | Hand-built | 10 | 10 | `jsonschema.validate` on parsed output |

Total: 60 train / 30 test.

## Episode contract

- Agent receives task text + task type + previous prompt (if any) + previous completion (if any) + previous reward
- Agent emits a **new full system prompt** (replace, not diff — simplest action space)
- Env runs LLM-under-test with `[system_prompt, user_task]` once
- Verifier returns 0/1 correctness + format/brevity bonuses
- Episode terminates when correctness == 1.0 OR `edit_turn >= 3`

## Reward (locked)

```
total = correctness + 0.1 * format_bonus + brevity_penalty
```
- `correctness ∈ {0, 1}` — programmatic verifier
- `format_bonus ∈ {0, 1}` — required tags present
- `brevity_penalty ∈ [-0.1, 0]` — only if prompt > 800 chars
- All components logged separately

## Compute budget

- Local smoke tests: CPU + small batches, Windows
- Full training: **HF Jobs `a10g-large`, ≤2h timeout**
- Demo: **HuggingFace Space, ZeroGPU**

## Out of scope (will not build)

- Multi-agent / hierarchical agents
- RAG, web search, tool use beyond verifier
- Persistent memory across episodes
- Custom reward model (we use programmatic verifiers)
- vLLM serving (transformers `generate()` is fine for 0.5B)
- Public Docker Space for env (in-process env in Gradio Space is enough)
- 4th task type (translation/summarization)
- 3B agent (only if Phase 5 has >2h slack)

## Submission targets

- HuggingFace Space: `<user>/promptops-arena`
- Model repo: `<user>/promptops-arena-agent` (LoRA adapter only)
- GitHub repo: public
- Video: ≤90 seconds, hosted on YouTube/Loom (UNLISTED), linked from README — never committed

## Judging weights

- Environment Innovation 40%
- Storytelling & Presentation 30%
- Showing Improvement in Rewards 20%
- Reward & Training Pipeline 10%

## Time gates (hard)

- T+24h: training must have started or drop to 2 task types
- T+36h: must have reward improvement; else ship "untrained agent vs zero-shot"
- T+44h: README + video + Space MUST be done; freeze features
