

def test_hedge_declines_detects_admission():
    from odistil.pipeline.hedge import declines

    assert declines("Reasoning...\nAnswer: I don't know")
    assert declines("Answer: I do not know who wrote it")
    assert declines("Answer: not sure")
    assert not declines("Answer: Charles Dickens")
    assert not declines("Answer: 1948")


def test_hedge_verifier_routes_through_declines():
    from odistil.pipeline.dataset import _verify

    dcfg = {"verify_mcq": True, "verify_qa": True, "verify_code": True}
    rec = {"kind": "hedge", "gold": ["dickens"], "answer": "Answer: I don't know"}
    assert _verify(rec, dcfg)[0]
    rec["answer"] = "Answer: Charles Dickens"
    assert not _verify(rec, dcfg)[0]


def test_hedge_question_recovered_from_qa_template():
    from odistil.pipeline.hedge import _question_of
    from odistil.pipeline.prompts import QA_TMPL

    q = "Who wrote Bleak House?"
    rec = {"prompt": QA_TMPL.format(question=q)}
    assert _question_of(rec) == q


def test_kodcode_normaliser_builds_runnable_tests():
    from odistil.pipeline.prompts import _normalize

    row = {
        "question": "Return the sum of a list.",
        "test": "from solution import total\n\ndef test_a():\n    assert total([1, 2]) == 3\n\ndef test_b():\n    assert total([]) == 0\n",
        "filter_reason": "",
        "benchmark_similarity": 0.5,
    }
    src = {"hf": "KodCode/KodCode-V1", "split": "train", "kind": "code", "_domain": "code"}
    rec = _normalize(src, row, 7)
    assert rec is not None
    assert "from solution import" not in rec["prompt"]
    assert rec["gold"]["tests"] == ["test_a()\ntest_b()"]

    from odistil.codeexec import check_with_tests

    g = rec["gold"]
    ok, _ = check_with_tests("```python\ndef total(xs): return sum(xs)\n```", "", g["tests"], preamble=g["preamble"])
    assert ok
    bad, _ = check_with_tests("```python\ndef total(xs): return 0\n```", "", g["tests"], preamble=g["preamble"])
    assert not bad
    # the model copying a test call into its own block must not be a NameError
    copied = "```python\ndef total(xs): return sum(xs)\ntest_a()\n```"
    ok2, err = check_with_tests(copied, "", g["tests"], preamble=g["preamble"])
    assert ok2, err


def test_kodcode_normaliser_skips_flagged_and_near_benchmark_rows():
    from odistil.pipeline.prompts import _normalize

    src = {"hf": "KodCode/KodCode-V1", "split": "train", "kind": "code", "_domain": "code"}
    base = {"question": "q", "test": "def test_x():\n    assert True\n", "filter_reason": "", "benchmark_similarity": 0.5}
    assert _normalize(src, {**base, "filter_reason": "dup"}, 1) is None
    assert _normalize(src, {**base, "benchmark_similarity": 0.97}, 2) is None
    assert _normalize(src, base, 3) is not None


def test_rollout_triage_splits_failures_and_same_size_control(tmp_path):
    import json

    from odistil.config import Config
    from odistil.pipeline.rollout import triage

    pool = tmp_path / "pool.jsonl"
    with pool.open("w") as f:
        for i in range(10):
            f.write(json.dumps({"id": f"p{i}", "domain": "knowledge", "kind": "mcq", "prompt": "x", "gold": "A"}) + "\n")
    out = tmp_path / "run"
    out.mkdir()
    with (out / "rollouts.jsonl").open("w") as f:
        for i in range(10):
            f.write(json.dumps({"id": f"p{i}", "domain": "knowledge", "kind": "mcq",
                                "correct": i % 3 != 0, "truncated": False, "tokens": 1, "why": ""}) + "\n")
    cfg = Config.load(None, "configs/v8.yaml")
    cfg.distill["rollout"]["pool"] = str(pool)
    cfg.distill["rollout"]["control_dir"] = str(tmp_path / "ctl")
    cfg.distill["out_dir"] = str(out)
    triage(cfg)
    failures = [json.loads(line)["id"] for line in (out / "prompts.jsonl").open()]
    control = [json.loads(line)["id"] for line in (tmp_path / "ctl" / "prompts.jsonl").open()]
    assert failures == ["p0", "p3", "p6", "p9"]
    assert len(control) == len(failures)
