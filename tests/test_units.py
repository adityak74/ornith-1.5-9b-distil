import json
import subprocess
import sys
from pathlib import Path

from odistil.codeexec import check_with_tests, extract_code
from odistil.config import Config
from odistil.mlxutil import opens_think, split_think
from odistil.textnorm import Decontaminator, final_answer, final_letter, qa_match

ROOT = Path(__file__).resolve().parents[1]


def test_final_letter_forms():
    assert final_letter("blah\nAnswer: C") == "C"
    assert final_letter("Answer: (B) because ...") == "B"
    assert final_letter("so the answer must be D") == "D"
    assert final_letter("Answer: Z", 4) is None


def test_final_answer_takes_last():
    assert final_answer("Answer: one\nmore\nAnswer: two") == "two"


def test_qa_match_normalizes():
    assert qa_match("Answer: The Paris.", ["paris", "paris france"])
    assert not qa_match("Answer: Berlin", ["paris"])


def test_decontaminator():
    d = Decontaminator(5)
    d.add("the quick brown fox jumps over the lazy dog")
    assert d.is_contaminated("well, the quick brown fox jumps over something")
    assert not d.is_contaminated("entirely unrelated sentence about ships")


def test_extract_code_prefers_longest_fence():
    text = "try\n```python\nx=1\n```\nor\n```python\ndef f():\n    return 2\n```"
    assert "def f()" in extract_code(text)


def test_code_verifier_accepts_and_rejects():
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    assert check_with_tests(good, "", ["assert add(1, 2) == 3"])[0]
    assert not check_with_tests(bad, "", ["assert add(1, 2) == 3"])[0]


def test_code_verifier_times_out():
    ok, err = check_with_tests("```python\nwhile True:\n    pass\n```", "", [], timeout=2)
    assert not ok and err == "timeout"


def test_config_routes_domains_to_teachers():
    cfg = Config.load()
    assert cfg.teacher_for("code")["name"] == "code"
    assert cfg.teacher_for("truthfulness")["name"] == "knowledge"
    assert cfg.model_ref("student:oq4").endswith("oQ4")


def test_cli_help_lists_all_stages():
    out = subprocess.run(
        [sys.executable, "-m", "odistil.cli", "--help"],
        capture_output=True, text=True, cwd=ROOT, check=True,
    ).stdout
    for stage in ["check", "prompts", "teach", "dataset", "train", "fuse", "quantize", "eval", "report"]:
        assert stage in out


def test_baseline_json_is_consistent():
    data = json.loads((ROOT / "benchmarks" / "baseline.json").read_text())
    for name, res in data["models"].items():
        for bench, r in res.items():
            assert abs(r["correct"] / r["total"] - r["accuracy"]) < 0.001, (name, bench)


def test_split_think_handles_template_opened_block():
    # template pre-opens <think>, so the generation only closes it
    assert split_think("reasoning here\n</think>\n\nAnswer: B") == ("Answer: B", "reasoning here")
    # a complete pair anywhere in the output
    assert split_think("<think>why</think>\nAnswer: C") == ("Answer: C", "why")
    # truncated mid-reasoning: no answer to score
    assert split_think("still thinking about it") == ("still thinking about it", None)
    assert split_think("<think>cut off") == ("", "cut off")
    # thinking disabled: template emits an empty block
    assert split_think("<think>\n\n</think>\n\nAnswer: A") == ("Answer: A", "")


def test_pre_opened_truncation_yields_no_answer():
    # prompt ended with '<think>\n' and the budget ran out mid-reasoning
    assert split_think("I should first consider", pre_opened=True) == ("", "I should first consider")
    # same text without a pre-opened block is a plain answer
    assert split_think("I should first consider") == ("I should first consider", None)


def test_opens_think_detects_template_state():
    assert opens_think("<|im_start|>assistant\n<think>\n")
    assert not opens_think("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert not opens_think("<|im_start|>assistant\n")
