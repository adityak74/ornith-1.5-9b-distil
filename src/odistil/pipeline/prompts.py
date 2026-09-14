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
import re
import string
from pathlib import Path

from ..config import Config

LETTERS = string.ascii_uppercase  # TruthfulQA MC1 goes up to 13 choices

TF_TMPL = (
    "Is the following statement true or false?\n\n{statement}\n\n"
    "Some statements that sound plausible are common misconceptions. Reason "
    "briefly, then end your reply with a final line of exactly 'Answer: True' "
    "or 'Answer: False'."
)
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
ABSTAIN_TMPL = (
    "Read the passage and answer the question.\n\n"
    "Passage: {context}\n\nQuestion: {question}\n\n"
    "If the passage does not contain the answer, say so plainly instead of "
    "guessing. End your reply with a final line of exactly "
    "'Answer: <answer>' or 'Answer: not stated in the passage'."
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
        elif hf.endswith("squad_v2"):
            # Only the unanswerable half. Every other slice in this pipeline
            # teaches the model to commit to an answer -- rejection sampling
            # keeps nothing else -- which is what cost v1 11 points of
            # abstention accuracy on TruthfulQA. See DECISIONS.md 28.
            if row["answers"]["text"]:
                return None
            prompt = ABSTAIN_TMPL.format(
                context=row["context"].strip()[:1200], question=row["question"].strip()
            )
            gold = "UNANSWERABLE"
        elif hf.endswith("misconceptions_tf"):
            # 'Correct' is 1.0 when the statement is true, 0.0 when it is a
            # misconception. Teaches resisting plausible falsehoods, which is
            # what TruthfulQA measures -- unlike recall data, which teaches the
            # opposite reflex. See DECISIONS.md 17.
            prompt = TF_TMPL.format(statement=row["Question"].strip())
            gold = "True" if float(row["Correct"]) == 1.0 else "False"
        elif hf.endswith("sciq"):
            choices = [row["distractor1"], row["distractor2"], row["distractor3"], row["correct_answer"]]
            random.shuffle(choices)
            prompt, gold = _mcq(row["question"], choices, choices.index(row["correct_answer"]))
        elif hf.endswith("trivia_qa"):
            aliases = row["answer"].get("normalized_aliases") or [row["answer"]["value"]]
            prompt = QA_TMPL.format(question=row["question"].strip())
            gold = list(aliases)
        elif hf.endswith("KodCode-V1"):
            # Fresh verifiable code for v8: MBPP is fully consumed (all 974
            # items across every split went into v1). KodCode ships pytest-style
            # tests that import from a `solution` module; the model's own code
            # defines that function in the same program, so drop the import and
            # append a runner that calls every test function. Rows the dataset
            # itself flags, and rows near-duplicating a benchmark, are skipped
            # -- the 13-gram decontamination in `dataset` still runs on top.
            if row.get("filter_reason") or float(row.get("benchmark_similarity") or 0) >= 0.95:
                return None
            test_src = "\n".join(
                ln for ln in row["test"].splitlines() if not ln.startswith("from solution import")
            )
            fns = re.findall(r"^def (test_\w+)\(", test_src, flags=re.MULTILINE)
            if not fns:
                return None
            runner = "\n".join(f"{fn}()" for fn in fns)
            prompt = CODE_TMPL.format(question=row["question"].strip(), tests=test_src.strip())
            gold = {"setup": "", "tests": [test_src, runner]}
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

    seed = cfg.distill["seed"]
    random.seed(seed)
    out = cfg.path("prompts.jsonl")
    n_written = 0

    # A fresh pool must not overlap an earlier run's prompts: ids encode the
    # source index, so any id the excluded file already holds is skipped.
    exclude: set[str] = set()
    for path in cfg.distill.get("exclude_prompts", []):
        with Path(path).open() as f:
            exclude |= {json.loads(line)["id"] for line in f if line.strip()}
    if exclude:
        print(f"  excluding {len(exclude)} ids already used")

    with out.open("w") as f:
        for domain, sources in cfg.distill["prompts"].items():
            if domains and domain not in domains:
                continue
            for src in sources:
                src = {**src, "_domain": domain}
                want = min(src["n"], limit) if limit else src["n"]
                print(f"  {domain:<13} {src['hf']}:{src['split']}  -> {want}")
                ds = load_dataset(src["hf"], src.get("config"), split=src["split"])
                # Seed per source: re-sizing one pool must not reshuffle another,
                # or traces already generated stop matching by id.
                #
                # A prefix-of-one-shuffle would additionally survive *growing* a
                # pool (1500 -> 2400 keeping the first 1500). Tried and reverted:
                # switching methods invalidated 2,400 traces already generated
                # under this one, which is 5 GPU-hours. Worth adopting at the
                # start of a run that has nothing to reuse.
                src_rng = random.Random(f"{seed}:{src['hf']}:{src['split']}")
                # oversample so exclusions and normaliser rejections still
                # leave `want` items, then stop once that many are written
                idxs = src_rng.sample(range(len(ds)), k=min(want * 2, len(ds)))
                got = 0
                for i in idxs:
                    if got >= want:
                        break
                    rec = _normalize(src, ds[i], i)
                    if rec and rec["id"] not in exclude:
                        f.write(json.dumps(rec) + "\n")
                        n_written += 1
                        got += 1
    print(f"wrote {n_written} prompts -> {out}")
    return out
