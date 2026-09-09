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
            if not isinstance(r, dict):   # free-text fields like "note"
                continue
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


def test_train_base_defaults_to_full_precision_student():
    cfg = Config.load(distill=ROOT / "configs" / "distill.yaml")
    assert cfg.train_base() == cfg.student_base()


def test_train_base_override_selects_a_quantized_checkpoint(tmp_path):
    """The QLoRA path stays available even though v3 does not use it: it was
    measured at 2.6x slower than bf16 here (DECISIONS.md 30)."""
    import yaml

    cfg = Config.load(distill=ROOT / "configs" / "v3.yaml")
    assert cfg.train_base() == cfg.student_base(), "v3 trains on bf16"

    d = yaml.safe_load((ROOT / "configs" / "v3.yaml").read_text())
    d["train"]["base"] = "student:oq4"
    p = tmp_path / "qlora.yaml"
    p.write_text(yaml.safe_dump(d))
    assert Config.load(distill=p).train_base().endswith("oQ4")


def test_v3_train_and_dataset_caps_agree():
    cfg = Config.load(distill=ROOT / "configs" / "v3.yaml")
    assert cfg.distill["train"]["max_seq_length"] == cfg.distill["dataset"]["max_seq_len"]


def test_truthfulqa_choices_are_shuffled():
    """The dataset lists the correct answer first in all 817 items; presenting
    them in that order makes the benchmark measure position bias."""
    import collections

    from odistil.eval.tasks import truthfulqa

    task = truthfulqa()
    gold = collections.Counter(i.gold for i in task.items)
    assert gold["A"] / len(task.items) < 0.4, "gold still concentrated on A"
    assert len(gold) > 4, "gold should span several positions"
    # deterministic across calls, so runs stay comparable
    assert [i.gold for i in truthfulqa().items] == [i.gold for i in task.items]


def test_abstention_slice_is_verified_by_declining():
    """v3's abstention data is the fix for DECISIONS.md 28: it must accept a
    refusal and reject a fabricated answer."""
    from odistil.pipeline.dataset import _verify

    cfg = {"verify_mcq": True, "verify_code": True, "verify_qa": True}
    rec = {"kind": "unanswerable", "gold": "UNANSWERABLE"}
    assert _verify({**rec, "answer": "Answer: not stated in the passage"}, cfg)[0]
    assert _verify({**rec, "answer": "The passage does not mention this."}, cfg)[0]
    assert not _verify({**rec, "answer": "Answer: Denver Broncos"}, cfg)[0]


def test_v3_mixture_reserves_room_for_abstention():
    cfg = Config.load(distill=ROOT / "configs" / "v3.yaml")
    mix = cfg.distill["dataset"]["mix"]
    assert 0 < mix["abstention"] <= 0.2, "too much refusal data would cost MMLU"
    assert abs(sum(mix.values()) - 1.0) < 1e-6


def test_chunkwise_gated_delta_matches_the_reference():
    """The chunkwise training path must agree with mlx-lm's sequential one,
    including in the model's real regime: unit-norm keys and strong decay."""
    import mlx.core as mx
    from mlx_lm.models.gated_delta import gated_delta_ops

    from odistil.gdn_chunkwise import gated_delta_chunkwise

    mx.random.seed(0)
    B, T, Hk, Dk, Hv, Dv = 1, 96, 2, 64, 4, 64
    inv = Dk**-0.5
    q = (inv**2) * mx.fast.rms_norm(mx.random.normal((B, T, Hk, Dk)), None, 1e-6)
    k = inv * mx.fast.rms_norm(mx.random.normal((B, T, Hk, Dk)), None, 1e-6)
    v = mx.random.normal((B, T, Hv, Dv))
    beta = mx.sigmoid(mx.random.normal((B, T, Hv)))
    s0 = mx.zeros((B, Hv, Dv, Dk))

    for gmean in (3.0, 0.0):          # weak and strong decay
        g = mx.sigmoid(mx.random.normal((B, T, Hv)) * 0.5 + gmean)
        yr, sr = gated_delta_ops(q, k, v, g, beta, s0)
        for chunk in (16, 64):        # 96 exercises the padded path at C=64
            yc, sc = gated_delta_chunkwise(q, k, v, g, beta, s0, chunk=chunk)
            scale = float(mx.abs(yr).max())
            assert float(mx.abs(yr - yc).max()) / scale < 1e-4
            assert float(mx.abs(sr - sc).max()) / float(mx.abs(sr).max()) < 1e-4


def test_chunkwise_gradients_match_the_reference():
    import mlx.core as mx
    from mlx_lm.models.gated_delta import gated_delta_ops

    from odistil.gdn_chunkwise import gated_delta_chunkwise

    mx.random.seed(1)
    B, T, Hk, Dk, Hv, Dv = 1, 64, 2, 32, 2, 32
    q = mx.random.normal((B, T, Hk, Dk)) * 0.3
    k = mx.random.normal((B, T, Hk, Dk)) * 0.3
    v = mx.random.normal((B, T, Hv, Dv))
    g = mx.sigmoid(mx.random.normal((B, T, Hv)) * 0.5 + 2.0)
    beta = mx.sigmoid(mx.random.normal((B, T, Hv)))
    s0 = mx.zeros((B, Hv, Dv, Dk))
    w = mx.random.normal((B, T, Hv, Dv))

    def make(fn, **kw):
        def loss(q, k, v, g, beta):
            y, s = fn(q, k, v, g, beta, s0, **kw)
            return (y * w).sum() + (s * s).sum()
        return loss

    args = (q, k, v, g, beta)
    ref = mx.grad(make(gated_delta_ops), argnums=(0, 1, 2, 3, 4))(*args)
    got = mx.grad(make(gated_delta_chunkwise, chunk=16), argnums=(0, 1, 2, 3, 4))(*args)
    for a, b in zip(ref, got):
        assert float(mx.abs(a - b).max()) / max(float(mx.abs(a).max()), 1e-9) < 1e-4


def test_unit_tri_inv_is_exact_and_differentiable():
    import mlx.core as mx

    from odistil.gdn_chunkwise import unit_tri_inv

    mx.random.seed(2)
    m = mx.eye(8) + mx.tril(mx.random.normal((3, 8, 8)) * 0.4, -1)
    assert float(mx.abs(unit_tri_inv(m) @ m - mx.eye(8)).max()) < 1e-5
    g = mx.grad(lambda x: unit_tri_inv(mx.eye(4) + mx.tril(x, -1)).sum())(mx.random.normal((4, 4)))
    mx.eval(g)
    assert not bool(mx.any(mx.isnan(g)))
