"""Stage 3b -- preference pairs for termination training (v10).

Every SFT run taught the student what a good trace looks like. None told it
what a bad one looks like, and the bad one it actually produces is specific:
a trace that reasons past the budget and never answers (DECISIONS.md 45).
v9 tried to fix that with positive examples alone and found too few (48).

This stage builds (chosen, rejected) pairs on the same prompt:

    chosen    a teacher trace that verified          -- teacher/*.jsonl of a run
    rejected  the student's own trace that did not   -- rollouts.jsonl of a run

The rejected side is on-policy: it is what the deployed oQ4 student wrote.
Most rejected traces are unterminated (truncated or harness-forced), so the
preference loss pushes probability from "keep reasoning" toward "stop and
answer" on exactly the prompts where the student does not.

A forced trace is stored with the harness sentence; that is stripped and the
block left unclosed, so the rejected text is what the student wrote before
the harness stepped in, not the harness's words.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import Config
from ..textnorm import Decontaminator
from .dataset import SYSTEM, _eval_texts, _target, _verify
from .rollout import strip_force


def _load(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def rejected_text(roll: dict) -> str:
    """The student's trace as it would have appeared with no harness."""
    think = strip_force(roll.get("think")) or ""
    if roll.get("truncated") or roll.get("forced"):
        return f"<think>\n{think}"                        # never closed
    return f"<think>\n{think}\n</think>\n\n{roll.get('answer') or ''}"


def build(cfg: Config, skip_decontam: bool = False) -> tuple[Path, Path]:
    pcfg = cfg.distill["pairs"]
    dcfg = cfg.distill["dataset"]
    keep_think = cfg.distill["teach"]["keep_think_in_target"]
    rng = random.Random(cfg.distill["seed"])

    decon = Decontaminator(dcfg["decontaminate_ngram"])
    if not skip_decontam:
        print("indexing eval sets for decontamination...")
        for t in _eval_texts():
            decon.add(t)

    from ..mlxutil import load_tokenizer

    tok = load_tokenizer(cfg.student_base())
    cap = dcfg["max_seq_len"]

    rolls: dict[str, dict] = {}
    for p in pcfg["rollouts"]:
        for r in _load(Path(p)):
            rolls.setdefault(r["id"], r)                  # first file wins
    teacher = []
    for p in pcfg["teacher"]:
        teacher += _load(Path(p))

    stats: dict[str, int] = {}
    pairs = []
    for t in teacher:
        s = rolls.get(t["id"])
        if s is None or s["correct"]:
            stats["no_failure"] = stats.get("no_failure", 0) + 1
            continue
        if not t.get("answer") or not _verify(t, dcfg)[0]:
            stats["teacher_unverified"] = stats.get("teacher_unverified", 0) + 1
            continue
        if not skip_decontam and decon.is_contaminated(t["prompt"]):
            stats["contaminated"] = stats.get("contaminated", 0) + 1
            continue
        chosen = _target(t, keep_think)
        if len(tok.encode(t["prompt"] + chosen)) > cap:
            stats["chosen_too_long"] = stats.get("chosen_too_long", 0) + 1
            continue
        unterminated = bool(s.get("truncated") or s.get("forced"))
        pairs.append({
            "id": t["id"], "domain": t["domain"], "kind": t["kind"],
            "prompt": t["prompt"], "chosen": chosen, "rejected": rejected_text(s),
            "rejected_unterminated": unterminated,
        })
        stats[f"kept:{t['kind']}"] = stats.get(f"kept:{t['kind']}", 0) + 1
        if unterminated:
            stats["kept:unterminated"] = stats.get("kept:unterminated", 0) + 1

    rng.shuffle(pairs)
    n_valid = max(int(len(pairs) * dcfg["valid_frac"]), 1) if pairs else 0
    splits = {"valid": pairs[:n_valid], "train": pairs[n_valid:]}
    paths = {}
    for split, recs in splits.items():
        p = cfg.path("train", f"pairs-{split}.jsonl")
        with p.open("w") as f:
            for r in recs:
                f.write(json.dumps({
                    "messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": r["prompt"]}],
                    "chosen": r["chosen"], "rejected": r["rejected"],
                    "id": r["id"], "kind": r["kind"],
                    "rejected_unterminated": r["rejected_unterminated"],
                }) + "\n")
        paths[split] = p
    stats["train"], stats["valid"] = len(splits["train"]), len(splits["valid"])
    with cfg.path("train", "pairs-stats.json").open("w") as f:
        json.dump(stats, f, indent=2)
    print("[pairs] " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    return paths["train"], paths["valid"]
