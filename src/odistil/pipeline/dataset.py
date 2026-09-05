"""Stage 3 -- verify, decontaminate, mix, and write the training files.

Rejection sampling is the whole point: a teacher trace is only worth training on
if its final answer is actually right. MCQ letters are matched against gold, QA
answers against the alias list, and code is executed against the dataset's own
tests. Anything that overlaps the eval sets by a 13-gram is dropped regardless.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from ..codeexec import check_with_tests, parses
from ..config import Config
from ..textnorm import Decontaminator, final_letter, qa_match

SYSTEM = "You are Ornith, a careful and concise assistant."


def _eval_texts() -> list[str]:
    """Every eval prompt we must never train on."""
    from datasets import load_dataset

    texts: list[str] = []
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    texts += [r["question"] for r in mmlu]
    tqa = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    texts += [r["question"] for r in tqa]
    he = load_dataset("openai/openai_humaneval", split="test")
    texts += [r["prompt"] for r in he]
    return texts


def _verify(rec: dict, dcfg: dict) -> tuple[bool, str]:
    kind, gold, answer = rec["kind"], rec["gold"], rec["answer"]
    if not answer.strip():
        return False, "empty"
    if kind == "mcq" and dcfg["verify_mcq"]:
        got = final_letter(answer)
        return (got == gold, f"mcq:{got}->{gold}")
    if kind == "qa" and dcfg["verify_qa"]:
        return (qa_match(answer, gold), "qa")
    if kind == "code" and dcfg["verify_code"]:
        ok, err = check_with_tests(answer, gold.get("setup", ""), gold["tests"])
        return ok, f"code:{err.splitlines()[-1][:60] if err else 'ok'}"
    if kind == "code_free" and dcfg["verify_code"]:
        # No tests exist for these. Require at least a syntactically valid code
        # block, which rejects truncated traces and prose-only answers.
        return parses(answer), "code_free:parse"
    return True, "unverified"


def _target(rec: dict, keep_think: bool) -> str:
    if keep_think and rec.get("think"):
        return f"<think>\n{rec['think']}\n</think>\n\n{rec['answer']}"
    return rec["answer"]


def build(cfg: Config, skip_decontam: bool = False) -> tuple[Path, Path]:
    dcfg = cfg.distill["dataset"]
    keep_think = cfg.distill["teach"]["keep_think_in_target"]
    rng = random.Random(cfg.distill["seed"])

    decon = Decontaminator(dcfg["decontaminate_ngram"])
    if not skip_decontam:
        print("indexing eval sets for decontamination...")
        for t in _eval_texts():
            decon.add(t)
        print(f"  {len(decon.index)} eval n-grams")

    kept: dict[str, list[dict]] = {}
    stats: Counter = Counter()
    for f in sorted((cfg.out_dir / "teacher").glob("*.jsonl")):
        with f.open() as fh:
            for line in fh:
                rec = json.loads(line)
                stats["seen"] += 1
                if not skip_decontam and decon.is_contaminated(rec["prompt"]):
                    stats["contaminated"] += 1
                    continue
                ok, why = _verify(rec, dcfg)
                if not ok:
                    stats[f"reject:{why.split(':')[0]}"] += 1
                    continue
                kept.setdefault(rec["domain"], []).append(rec)
                stats["kept"] += 1

    # Rebalance to the configured mixture, capped by the scarcest domain.
    mix = dcfg["mix"]
    budget = min(
        (len(v) / mix[k] for k, v in kept.items() if mix.get(k)),
        default=0,
    )
    samples: list[dict] = []
    for domain, recs in kept.items():
        n = int(budget * mix.get(domain, 0))
        rng.shuffle(recs)
        samples += recs[:n]
        stats[f"mix:{domain}"] = n
    rng.shuffle(samples)

    n_valid = max(int(len(samples) * dcfg["valid_frac"]), 1) if samples else 0
    splits = {"valid": samples[:n_valid], "train": samples[n_valid:]}

    paths = {}
    for split, recs in splits.items():
        p = cfg.path("train", f"{split}.jsonl")
        with p.open("w") as f:
            for r in recs:
                f.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "system", "content": SYSTEM},
                                {"role": "user", "content": r["prompt"]},
                                {"role": "assistant", "content": _target(r, keep_think)},
                            ]
                        }
                    )
                    + "\n"
                )
        paths[split] = p

    with cfg.path("train", "stats.json").open("w") as f:
        json.dump(dict(stats), f, indent=2)
    print(json.dumps(dict(stats), indent=2))
    print(f"train={len(splits['train'])} valid={len(splits['valid'])} -> {paths['train'].parent}")
    return paths["train"], paths["valid"]
