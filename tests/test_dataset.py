

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


def test_unclosed_think_detection():
    from odistil.mlxutil import FORCE_STR, _unclosed, split_think

    assert _unclosed("some reasoning with no close", pre_opened=True)
    assert not _unclosed("reasoning</think>\nAnswer: B", pre_opened=True)
    assert _unclosed("<think>opened here", pre_opened=False)
    assert not _unclosed("plain answer", pre_opened=False)
    # a forced continuation parses as a real answer, not a truncation
    raw = "half a thought" + FORCE_STR + "Answer: C"
    answer, think = split_think(raw, pre_opened=True)
    assert answer == "Answer: C"
    assert think.startswith("half a thought")


def test_rollout_harvest_keeps_correct_rows_and_strips_force_sentence(tmp_path):
    import json

    from odistil.config import Config
    from odistil.mlxutil import FORCE_SENTENCE
    from odistil.pipeline.rollout import harvest, strip_force

    pool = tmp_path / "pool.jsonl"
    with pool.open("w") as f:
        for i in range(4):
            f.write(json.dumps({"id": f"p{i}", "domain": "knowledge", "kind": "mcq", "prompt": "x", "gold": "A"}) + "\n")
    out = tmp_path / "run"
    out.mkdir()
    rolls = [  # p0 natural-correct, p1 forced-correct, p2 wrong, p3 forced-wrong
        {"id": "p0", "correct": True, "forced": False, "think": "reasoning", "answer": "A"},
        {"id": "p1", "correct": True, "forced": True, "think": "cut off\n\n" + FORCE_SENTENCE, "answer": "A"},
        {"id": "p2", "correct": False, "forced": False, "think": "r", "answer": "B"},
        {"id": "p3", "correct": False, "forced": True, "think": "r", "answer": "B"},
    ]
    with (out / "rollouts.jsonl").open("w") as f:
        for r in rolls:
            f.write(json.dumps({**r, "domain": "knowledge", "kind": "mcq", "truncated": False, "tokens": 1, "why": ""}) + "\n")
    extra = tmp_path / "code.jsonl"
    extra.write_text(json.dumps({"id": "t0", "domain": "code", "kind": "code", "prompt": "y", "gold": {}, "answer": "z"}) + "\n")

    cfg = Config.load(None, "configs/v9.yaml")
    cfg.distill["out_dir"] = str(out)
    cfg.distill["rollout"]["pool"] = str(pool)
    cfg.distill["rollout"]["harvest"] = {"forced": True, "include_teacher": [str(extra)]}
    harvest(cfg)
    rows = {json.loads(line)["id"]: json.loads(line) for line in (out / "teacher" / "self.jsonl").open()}
    assert set(rows) == {"p0", "p1"}
    assert rows["p1"]["forced"] is True and rows["p1"]["think"] == "cut off"
    assert rows["p0"]["forced"] is False and rows["p0"]["gold"] == "A"
    assert (out / "teacher" / "code.jsonl").exists()

    cfg.distill["rollout"]["harvest"] = {"forced": False}
    harvest(cfg)
    rows = [json.loads(line)["id"] for line in (out / "teacher" / "self.jsonl").open()]
    assert rows == ["p0"]
    assert strip_force(None) is None
    assert strip_force("plain") == "plain"


def test_pairs_rejected_text_and_build(tmp_path):
    import json

    from odistil.config import Config
    from odistil.mlxutil import FORCE_SENTENCE
    from odistil.pipeline.pairs import build, rejected_text

    assert rejected_text({"think": "r", "answer": "B", "truncated": False, "forced": False}) == "<think>\nr\n</think>\n\nB"
    assert rejected_text({"think": "r\n\n" + FORCE_SENTENCE, "answer": "B", "forced": True}) == "<think>\nr"
    assert rejected_text({"think": "r", "answer": "", "truncated": True}) == "<think>\nr"

    teacher = tmp_path / "t.jsonl"
    with teacher.open("w") as f:
        for i, ans in enumerate(["A", "B", "A"]):   # p1's teacher is wrong -> unverified
            f.write(json.dumps({"id": f"p{i}", "domain": "knowledge", "kind": "mcq", "prompt": f"q{i}",
                                "gold": "A", "think": "t", "answer": ans}) + "\n")
    rolls = tmp_path / "r.jsonl"
    with rolls.open("w") as f:
        f.write(json.dumps({"id": "p0", "correct": False, "truncated": True, "think": "x", "answer": ""}) + "\n")
        f.write(json.dumps({"id": "p1", "correct": False, "truncated": True, "think": "x", "answer": ""}) + "\n")
        f.write(json.dumps({"id": "p2", "correct": True, "truncated": False, "think": "x", "answer": "A"}) + "\n")
    cfg = Config.load(None, "configs/v10.yaml")
    cfg.distill["out_dir"] = str(tmp_path / "run")
    cfg.distill["pairs"] = {"teacher": [str(teacher)], "rollouts": [str(rolls)]}
    cfg.distill["dataset"]["valid_frac"] = 0.0
    train, valid = build(cfg, skip_decontam=True)
    rows = [json.loads(line) for line in train.open()] + [json.loads(line) for line in valid.open()]
    assert [r["id"] for r in rows] == ["p0"]
    assert rows[0]["rejected"] == "<think>\nx" and rows[0]["rejected_unterminated"] is True
    assert rows[0]["chosen"].startswith("<think>\nt\n</think>")
    stats = json.loads((tmp_path / "run" / "train" / "pairs-stats.json").read_text())
    assert stats["teacher_unverified"] == 1 and stats["no_failure"] == 1
