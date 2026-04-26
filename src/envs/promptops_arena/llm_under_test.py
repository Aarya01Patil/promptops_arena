"""
Frozen LLM-under-test. The agent's prompts are evaluated by running this model.

Two backends:
- "transformers": real Qwen2.5-0.5B-Instruct via HF transformers
- "stub": deterministic stub for fast local CI / smoke tests

Selected via env var PROMPTOPS_LLM_BACKEND (default: "stub" if torch unavailable).
The stub recognizes a few hand-written "good" prompt patterns to give the
env smoke test something non-zero to chew on.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Optional


_DEFAULT_MODEL = os.environ.get("PROMPTOPS_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
_BACKEND = os.environ.get("PROMPTOPS_LLM_BACKEND", "auto").lower()
_MAX_NEW_TOKENS = int(os.environ.get("PROMPTOPS_LLM_MAX_NEW_TOKENS", "256"))


class _StubBackend:
    """
    Deterministic stub. Reads the system prompt + task, and produces a
    plausible-looking completion that the verifiers can sometimes pass
    when the prompt asks for the right format.

    Heuristic logic:
    - If prompt mentions <answer> tags, wrap a guessed answer in them
    - For math: try to compute the answer naively (look for numbers in question)
    - For code: emit a trivial function that returns 0 (will fail tests)
    - For JSON: emit an empty object

    This means: with a good prompt, math gets ~30% by luck; code/json need
    a real LLM. That's fine — stub is only for plumbing tests.
    """

    name = "stub"

    def generate(self, system_prompt: str, user_task: str) -> str:
        sp = (system_prompt or "").lower()
        ut = (user_task or "")

        # Order matters: JSON first, then code, then math (most specific to least)
        wants_json = "json" in sp
        wants_code = ("python" in sp or "function" in sp) and "json" not in sp
        wants_answer_tag = "<answer>" in (system_prompt or "")
        wants_boxed = "boxed" in sp

        if wants_json:
            return "```json\n{}\n```"

        if wants_code:
            body = "def solve(*a, **k):\n    return 0\n"
            return f"```python\n{body}```"

        nums = re.findall(r"-?\d+(?:\.\d+)?", ut)
        guess = nums[-1] if nums else "0"

        if wants_answer_tag:
            return f"Working...\n<answer>{guess}</answer>"
        if wants_boxed:
            return f"Working...\n\\boxed{{{guess}}}"
        return f"The answer is {guess}."


class _TransformersBackend:
    name = "transformers"

    def __init__(self, model_id: str = _DEFAULT_MODEL):
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        self._torch = torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device,
        )
        self.model.eval()
        self.device = device

    def generate(self, system_prompt: str, user_task: str) -> str:
        msgs = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": user_task},
        ]
        encoded = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt",
        )
        # apply_chat_template may return a Tensor (older transformers) or a
        # BatchEncoding/dict (newer); normalize to input_ids tensor.
        if hasattr(encoded, "input_ids"):
            input_ids = encoded.input_ids
        elif isinstance(encoded, dict):
            input_ids = encoded["input_ids"]
        else:
            input_ids = encoded
        input_ids = input_ids.to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        return text


_lock = threading.Lock()
_backend_singleton: Optional[object] = None


def _select_backend() -> object:
    global _backend_singleton
    with _lock:
        if _backend_singleton is not None:
            return _backend_singleton
        choice = _BACKEND
        if choice == "auto":
            try:
                import torch  # noqa: F401
                import transformers  # noqa: F401
                choice = "transformers"
            except ImportError:
                choice = "stub"
        if choice == "transformers":
            try:
                _backend_singleton = _TransformersBackend()
            except Exception as e:
                print(f"[llm_under_test] transformers backend failed ({e}); falling back to stub")
                _backend_singleton = _StubBackend()
        else:
            _backend_singleton = _StubBackend()
        return _backend_singleton


def generate(system_prompt: str, user_task: str) -> str:
    """Run the frozen LLM-under-test. Threadsafe singleton."""
    backend = _select_backend()
    return backend.generate(system_prompt, user_task)


def backend_name() -> str:
    return _select_backend().name  # type: ignore[attr-defined]
