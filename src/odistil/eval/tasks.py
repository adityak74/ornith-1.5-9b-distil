"""Eval tasks: MMLU (sampled), TruthfulQA MC1 (full), HumanEval (full).

Kept deliberately close to the baseline table this project is measured against:
a fixed 1000-item MMLU sample, all 817 TruthfulQA MC1 items, all 164 HumanEval
problems, thinking enabled, greedy decoding.
"""

from __future__ import annotations

import random
import string
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..codeexec import run_program
from ..textnorm import final_letter

LETTERS = string.ascii_uppercase  # TruthfulQA MC1 goes up to 13 choices

MCQ_TMPL = (
    "{question}\n\n{choices}\n\n"
    "Reason briefly, then end with a final line of exactly 'Answer: <letter>'."
)
CODE_TMPL = (
    "Complete the following Python function. Return the full function, including "
    "the signature, in a single ```python block. Do not add example usage.\n\n"
    "```python\n{prompt}```"
)


@dataclass
class Item:
    id: str
    prompt: str
    gold: Any
    meta: dict = field(default_factory=dict)


@dataclass
class Task:
    name: str
    items: list[Item]
    score: Callable[[Item, str], bool]
    max_tokens: int = 2048


def _mcq(question: str, choices: list[str]) -> str:
    body = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return MCQ_TMPL.format(question=question.strip(), choices=body)


def mmlu(sample: int | None = 1000, seed: int = 17) -> Task:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    idxs = list(range(len(ds)))
    if sample and sample < len(ds):
        idxs = random.Random(seed).sample(idxs, sample)
        idxs.sort()
    items = [
        Item(
            id=f"mmlu-{i}",
            prompt=_mcq(ds[i]["question"], list(ds[i]["choices"])),
            gold=LETTERS[int(ds[i]["answer"])],
            meta={"subject": ds[i]["subject"]},
        )
        for i in idxs
    ]
    return Task("mmlu", items, lambda it, out: final_letter(out, 4) == it.gold)


def truthfulqa(seed: int = 17) -> Task:
    """TruthfulQA MC1, with the choices shuffled.

    The dataset lists the correct answer **first** in every one of the 817
    items. Presenting them in dataset order makes 'A' the gold answer every
    time, so the score largely measures position bias -- a model that always
    answers 'A' scores 100%. Choices are shuffled with a per-item seed, which
    is deterministic across runs and models so comparisons stay valid.
    """
    from datasets import load_dataset

    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    items = []
    for i, row in enumerate(ds):
        choices = list(row["mc1_targets"]["choices"])
        correct = choices[list(row["mc1_targets"]["labels"]).index(1)]
        order = list(range(len(choices)))
        random.Random(f"{seed}:tqa:{i}").shuffle(order)
        shuffled = [choices[j] for j in order]
        gold = LETTERS[shuffled.index(correct)]
        items.append(Item(f"tqa-{i}", _mcq(row["question"], shuffled), gold,
                          {"n_choices": len(choices)}))
    return Task(
        "truthfulqa",
        items,
        lambda it, out: final_letter(out, it.meta["n_choices"]) == it.gold,
    )


def _he_score(item: Item, out: str) -> bool:
    """Run the model's code against the problem's tests.

    The original prompt is prepended as a preamble: it carries the imports and
    any helper the problem defines (`encode_cyclic`, `encode_shift`), which a
    chat model naturally omits when asked for "the function". Without it, three
    correct solutions failed with NameError. The model's own definition comes
    after the stub, so it wins.
    """
    from ..codeexec import extract_code

    program = "\n".join(
        [
            item.gold["prompt"],
            extract_code(out),
            item.gold["test"],
            f"check({item.gold['entry_point']})",
        ]
    )
    ok, _ = run_program(program, timeout=15.0)
    return ok


def humaneval() -> Task:
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    items = [
        Item(
            row["task_id"],
            CODE_TMPL.format(prompt=row["prompt"]),
            {"test": row["test"], "entry_point": row["entry_point"], "prompt": row["prompt"]},
        )
        for row in ds
    ]
    return Task("humaneval", items, _he_score)


REGISTRY = {"mmlu": mmlu, "truthfulqa": truthfulqa, "humaneval": humaneval}
