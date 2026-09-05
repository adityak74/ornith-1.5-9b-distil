"""Stage 1 -- build the prompt pool.

Pulls the source datasets named in configs/distill.yaml and normalizes every
item into a single record shape:

    {id, domain, kind, prompt, gold, meta}

`gold` is what stage 3 verifies the teacher against: a letter for mcq, an alias
list for qa, a test program for code, and None for free-form code instructions.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import Config

LETTERS = "ABCDEFGHIJ"

MCQ_TMPL = (
    "{question}\n\n{choices}\n\n"
    "Think it through, then end your reply with a final line of exactly "
    "'Answer: <letter>'."
)
QA_TMPL = (
    "{question}\n\n"
    "Answer accurately and concisely. If you are not confident, say so rather than guessing. "
    "End your reply with a final line of exactly 'Answer: <answer>'."
)
CODE_TMPL = (
    "{question}\n\n"
    "Write a complete Python solution. Put the final code in a single ```python block. "
    "Your function must satisfy:\n{tests}"
)


def _mcq(question: str, choices: list[str], answer_idx: int) -> tuple[str, str]:
    body = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return MCQ_TMPL.format(question=question.strip(), choices=body), LETTERS[answer_idx]


def _normalize(src: dict, row: dict, idx: int) -> dict | None:
    hf, kind = src["hf"], src["kind"]
    try:
        if hf.endswith("mmlu"):
            prompt, gold = _mcq(row["question"], list(row["choices"]), int(row["answer"]))
        elif hf.endswith("ai2_arc"):
            ch = row["choices"]
            labels = list(ch["label"])
            if row["answerKey"] not in labels:
                return None
            prompt, gold = _mcq(row["question"], list(ch["text"]), labels.index(row["answerKey"]))
        elif hf.endswith("openbookqa"):
            ch = row["choices"]
            labels = list(ch["label"])
            if row["answerKey"] not in labels:
                return None
            prompt, gold = _mcq(row["question_stem"], list(ch["text"]), labels.index(row["answerKey"]))
        elif hf.endswith("sciq"):
            choices = [row["distractor1"], row["distractor2"], row["distractor3"], row["correct_answer"]]
            random.shuffle(choices)
            prompt, gold = _mcq(row["question"], choices, choices.index(row["correct_answer"]))
        elif hf.endswith("trivia_qa"):
            aliases = row["answer"].get("normalized_aliases") or [row["answer"]["value"]]
            prompt = QA_TMPL.format(question=row["question"].strip())
            gold = list(aliases)
        elif hf.endswith("mbpp"):
            tests = "\n".join(row["test_list"])
            prompt = CODE_TMPL.format(question=row["text"].strip(), tests=tests)
            gold = {"setup": row.get("test_setup_code", ""), "tests": list(row["test_list"])}
        else:  # free-form instruction tuning data (no verifier)
            prompt, gold = row.get("query") or row.get("instruction") or row.get("prompt"), None
            if not prompt:
                return None
    except (KeyError, ValueError, IndexError):
        return None
    return {
        "id": f"{hf.split('/')[-1]}-{src.get('config', 'default')}-{src['split']}-{idx}",
        "domain": src["_domain"],
        "kind": kind,
        "prompt": prompt,
        "gold": gold,
        "meta": {"source": hf, "config": src.get("config"), "split": src["split"]},
    }


def build(cfg: Config, domains: list[str] | None = None, limit: int | None = None) -> Path:
    from datasets import load_dataset

    rng = random.Random(cfg.distill["seed"])
    random.seed(cfg.distill["seed"])
    out = cfg.path("prompts.jsonl")
    n_written = 0

    with out.open("w") as f:
        for domain, sources in cfg.distill["prompts"].items():
            if domains and domain not in domains:
                continue
            for src in sources:
                src = {**src, "_domain": domain}
                want = min(src["n"], limit) if limit else src["n"]
                print(f"  {domain:<13} {src['hf']}:{src['split']}  -> {want}")
                ds = load_dataset(src["hf"], src.get("config"), split=src["split"])
                idxs = rng.sample(range(len(ds)), k=min(want, len(ds)))
                for i in idxs:
                    rec = _normalize(src, ds[i], i)
                    if rec:
                        f.write(json.dumps(rec) + "\n")
                        n_written += 1
    print(f"wrote {n_written} prompts -> {out}")
    return out
