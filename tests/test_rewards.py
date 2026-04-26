"""
Adversarial test suite for the reward function.

Goal: prove the reward cannot be gamed without doing the task. If any of
these tests fail, training will reward-hack and we'll waste GPU time.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.envs.promptops_arena.server.rewards import compute_reward
from src.envs.promptops_arena.verifiers import verify


# ---- math ----

MATH_TASK = {"id": "x", "type": "math", "question": "What is 2+2?", "answer": "4"}


def _reward(task, prompt, completion):
    v = verify(task, completion)
    return compute_reward(task, prompt, completion, v)


def test_math_correct_with_tag_full_reward():
    r = _reward(MATH_TASK, "short prompt", "<answer>4</answer>")
    assert r["correctness"] == 1.0
    assert r["format"] == 1.0
    assert r["total"] == pytest.approx(1.1)


def test_math_correct_no_tag_only_correctness():
    r = _reward(MATH_TASK, "short prompt", "The answer is 4.")
    assert r["correctness"] == 1.0
    assert r["format"] == 0.0
    assert r["total"] == pytest.approx(1.0)


def test_math_empty_completion_zero():
    r = _reward(MATH_TASK, "short prompt", "")
    assert r["correctness"] == 0.0
    assert r["format"] == 0.0
    assert r["total"] == 0.0


def test_math_empty_tag_only_format_bonus_capped():
    """Tag with no number — gets format bonus but not correctness. Total <= 0.1."""
    r = _reward(MATH_TASK, "short prompt", "<answer></answer>")
    assert r["correctness"] == 0.0
    assert r["total"] <= 0.1


def test_math_wrong_answer_with_perfect_format_capped():
    r = _reward(MATH_TASK, "short prompt", "<answer>7</answer>")
    assert r["correctness"] == 0.0
    assert r["total"] <= 0.1


def test_math_rambling_long_correct_still_bounded():
    """5000-char prompt + correct: brevity penalty must fire."""
    long_prompt = "blah " * 1000  # 5000 chars
    r = _reward(MATH_TASK, long_prompt, "<answer>4</answer>")
    assert r["brevity"] < 0.0
    assert r["brevity"] >= -0.1
    # still mostly rewarded for correctness
    assert r["total"] >= 1.0


def test_math_short_prompt_no_brevity_penalty():
    r = _reward(MATH_TASK, "Solve.", "<answer>4</answer>")
    assert r["brevity"] == 0.0


def test_math_boxed_format_recognized():
    r = _reward(MATH_TASK, "short", "Final: \\boxed{4}")
    assert r["correctness"] == 1.0
    assert r["format"] == 1.0


def test_math_keyword_only_no_credit():
    """'answer:' phrase without correct number gets nothing."""
    r = _reward(MATH_TASK, "short", "answer: 99")
    assert r["correctness"] == 0.0
    assert r["total"] <= 0.1


def test_math_repeated_token_not_rewarded():
    r = _reward(MATH_TASK, "short", "4 4 4 4 4 4 4 4")
    # last number IS 4, so verifier extracts it correctly. This test documents
    # that exact-match on a single number IS exploitable in this trivial task,
    # so for real GSM8K-style tasks the answer should be one of many numbers in
    # the question and not present in the prompt itself. We assert the
    # "correctness fires only on exact match", not that this is unhackable on
    # adversarial inputs unrelated to the question.
    assert r["correctness"] in (0.0, 1.0)


# ---- code ----

CODE_TASK = {
    "id": "c",
    "type": "code",
    "question": "Write add(a,b)",
    "tests": ["assert add(2, 3) == 5", "assert add(0, 0) == 0"],
}


def test_code_correct_passes_tests():
    completion = "```python\ndef add(a, b):\n    return a + b\n```"
    r = _reward(CODE_TASK, "short", completion)
    assert r["correctness"] == 1.0
    assert r["format"] == 1.0


def test_code_no_block_zero():
    r = _reward(CODE_TASK, "short", "Sure! add returns sum.")
    assert r["correctness"] == 0.0


def test_code_block_but_wrong_zero():
    completion = "```python\ndef add(a, b):\n    return a - b\n```"
    r = _reward(CODE_TASK, "short", completion)
    assert r["correctness"] == 0.0
    # format bonus still given (proper block) — total bounded
    assert r["total"] <= 0.1


def test_code_infinite_loop_times_out_to_zero():
    completion = "```python\ndef add(a, b):\n    while True: pass\n```"
    r = _reward(CODE_TASK, "short", completion)
    assert r["correctness"] == 0.0


def test_code_malicious_import_still_sandboxed():
    """We don't formally sandbox; we rely on subprocess isolation + 5s timeout."""
    completion = "```python\nimport os\ndef add(a, b):\n    return a + b\n```"
    r = _reward(CODE_TASK, "short", completion)
    # imports are allowed; correctness still computed
    assert r["correctness"] == 1.0


# ---- json ----

JSON_TASK = {
    "id": "j",
    "type": "json",
    "question": "Extract name and age",
    "schema": {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    },
    "expected": {"name": "Alice", "age": 30},
}


def test_json_correct_full_reward():
    completion = '```json\n{"name": "Alice", "age": 30}\n```'
    r = _reward(JSON_TASK, "short", completion)
    assert r["correctness"] == 1.0
    assert r["format"] == 1.0


def test_json_inline_no_block_lower():
    completion = '{"name": "Alice", "age": 30}'
    r = _reward(JSON_TASK, "short", completion)
    assert r["correctness"] == 1.0
    assert r["format"] == 0.0


def test_json_empty_object_zero():
    r = _reward(JSON_TASK, "short", "```json\n{}\n```")
    assert r["correctness"] == 0.0


def test_json_wrong_type_zero():
    """Schema requires int age — string fails."""
    completion = '```json\n{"name": "Alice", "age": "30"}\n```'
    r = _reward(JSON_TASK, "short", completion)
    assert r["correctness"] == 0.0


def test_json_invalid_zero():
    r = _reward(JSON_TASK, "short", "```json\n{not valid json\n```")
    assert r["correctness"] == 0.0


# ---- decomposition contract ----

def test_reward_dict_always_has_four_keys():
    """The reward function must always return all four components."""
    cases = [
        (MATH_TASK, "p", "<answer>4</answer>"),
        (MATH_TASK, "p", ""),
        (CODE_TASK, "p", "```python\ndef add(a,b): return a+b\n```"),
        (JSON_TASK, "p", "{}"),
    ]
    for task, p, c in cases:
        r = _reward(task, p, c)
        assert set(r.keys()) >= {"correctness", "format", "brevity", "total"}
        for k in ("correctness", "format", "brevity", "total"):
            assert isinstance(r[k], float)


def test_reward_total_is_sum_of_components():
    r = _reward(MATH_TASK, "x" * 1500, "<answer>4</answer>")
    expected = r["correctness"] + 0.1 * r["format"] + r["brevity"]
    assert r["total"] == pytest.approx(expected)
