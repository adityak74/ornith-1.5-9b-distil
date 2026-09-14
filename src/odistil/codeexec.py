"""Extract python code from model output and run it in a throwaway subprocess.

Untrusted model code: always a separate process, hard timeout, temp cwd.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    blocks = FENCE.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    # No fence: keep the tail starting at the first def/import/class.
    m = re.search(r"^(?:from |import |def |class )", text, re.MULTILINE)
    return text[m.start():].strip() if m else text.strip()


def parses(text: str) -> bool:
    """Cheapest possible filter: is there extractable code that is valid Python?

    Used for free-form coding instructions, which have no tests to run against.
    Catches truncated traces and prose-only answers without executing anything.
    """
    import ast

    code = extract_code(text)
    if not code.strip():
        return False
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def run_program(program: str, timeout: float = 10.0) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "prog.py"
        f.write_text(program)
        try:
            p = subprocess.run(
                [sys.executable, str(f)],
                cwd=td,
                capture_output=True,
                check=False,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        return p.returncode == 0, (p.stderr or "")[-2000:]


def check_with_tests(
    completion: str, setup: str, tests: list[str], timeout: float = 10.0, preamble: str = ""
) -> tuple[bool, str]:
    # `preamble` runs BEFORE the model's code. Test-function definitions go
    # there: the prompt shows the model named tests, and it sometimes copies a
    # call like `test_x()` into its own block, which is a NameError if the
    # definitions only arrive afterwards. Definitions are inert until called,
    # so hoisting them is safe; bare asserts (MBPP) must stay after the code.
    program = "\n".join([preamble, extract_code(completion), setup, *tests])
    return run_program(program, timeout)
