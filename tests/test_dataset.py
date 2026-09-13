

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
