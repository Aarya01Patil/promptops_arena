from .math_verifier import verify_math
from .code_verifier import verify_code
from .json_verifier import verify_json

__all__ = ["verify_math", "verify_code", "verify_json", "verify"]


def verify(task: dict, completion: str) -> dict:
    """
    Dispatch to the right verifier based on task['type'].

    Returns a dict: {correctness: 0.0|1.0, format_ok: bool, details: str}
    """
    task_type = task.get("type", "")
    if task_type == "math":
        return verify_math(task, completion)
    if task_type == "code":
        return verify_code(task, completion)
    if task_type == "json":
        return verify_json(task, completion)
    return {"correctness": 0.0, "format_ok": False, "details": f"Unknown task type: {task_type}"}
