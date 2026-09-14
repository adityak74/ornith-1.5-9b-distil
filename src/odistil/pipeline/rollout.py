"""Stage 1b -- roll the student out and keep what it gets wrong.

Every run v1-v7 trained on a *static* prompt pool: the teacher answered, the
verifier filtered, the student imitated. That never asks what this particular
student already knows. Most of a static pool is spent re-teaching things the
student answers correctly on its own.

This stage runs the current student over a fresh pool, verifies every answer
with the same verifiers the dataset stage uses, and keeps only the prompts it
failed. The teacher is then asked only about those. The training set becomes a
function of the student's errors rather than of the pool's composition --
DAgger's insight, applied at the prompt level (arXiv:2306.13649 applies it at
the token level; that is the natural next step once this one is measured).

Two arms are written, same size, for a controlled comparison:

    prompts.jsonl            the failures         -> v8
    <control_dir>/prompts.jsonl   a random subset  -> v8-control

Without the control, a gain from error-conditioned data is confounded with
"more training"; with it, the only difference between arms is *which* prompts.

The student is rolled out at its deployed bit width. Its oQ4 failures include
the ones quantization introduced, which are exactly the ones worth repairing.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import Config
from ..mlxutil import generate_batch
from .dataset import _verify
from .teach import SYSTEM


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def _budget(rcfg: dict, kind: str) -> int:
    budgets = rcfg.get("max_tokens", {})
    return budgets.get(kind, budgets.get("default", 2048))


def run(cfg: Config, limit: int | None = None) -> Path:
    rcfg = cfg.distill["rollout"]
    pool = Path(rcfg["pool"])
    student = cfg.model_ref(rcfg["student"])
    out = cfg.path("rollouts.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    with pool.open() as f:
        prompts = [json.loads(line) for line in f if line.strip()]
    done = _done_ids(out)
    todo = [p for p in prompts if p["id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"[rollout] {student}: {len(todo)} to run ({len(done)} cached) from {pool}")

    bs = rcfg.get("batch_size", 8)
    dcfg = cfg.distill["dataset"]
    # batch by kind so one budget applies per batch
    todo.sort(key=lambda p: p["kind"])
    with out.open("a") as f:
        for i in range(0, len(todo), bs):
            chunk = todo[i : i + bs]
            chunk = [c for c in chunk if c["kind"] == chunk[0]["kind"]]
            comps = generate_batch(
                student,
                [c["prompt"] for c in chunk],
                system=SYSTEM,
                max_tokens=_budget(rcfg, chunk[0]["kind"]),
                temp=0.0,
                think=True,
                batch_size=bs,
            )
            for rec, comp in zip(chunk, comps, strict=True):
                ok, why = _verify({**rec, "answer": comp.text}, dcfg)
                f.write(json.dumps({
                    "id": rec["id"],
                    "domain": rec["domain"],
                    "kind": rec["kind"],
                    "correct": bool(ok),
                    "truncated": comp.truncated,
                    "tokens": comp.tokens,
                    "why": why,
                    # kept so a verifier fix can re-score without regenerating
                    "answer": comp.text,
                }) + "\n")
                f.flush()
            n = min(i + bs, len(todo))
            print(f"  {n}/{len(todo)}", end="\r", flush=True)

    return triage(cfg)


def triage(cfg: Config) -> Path:
    """Split the pool into the failure arm and a same-size random control."""
    rcfg = cfg.distill["rollout"]
    pool = Path(rcfg["pool"])
    with pool.open() as f:
        prompts = {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}
    with cfg.path("rollouts.jsonl").open() as f:
        rolls = [json.loads(line) for line in f if line.strip()]

    by_kind: dict[str, dict[str, int]] = {}
    failed = []
    for r in rolls:
        k = by_kind.setdefault(r["kind"], {"n": 0, "wrong": 0, "truncated": 0})
        k["n"] += 1
        if not r["correct"]:
            k["wrong"] += 1
            failed.append(r["id"])
        if r["truncated"]:
            k["truncated"] += 1

    print("\n[rollout] student accuracy on the fresh pool:")
    for kind, k in sorted(by_kind.items()):
        acc = 1 - k["wrong"] / max(k["n"], 1)
        print(f"  {kind:<10} {acc:6.1%} correct   {k['wrong']:4d} wrong   {k['truncated']:4d} truncated   (n={k['n']})")

    failure_arm = cfg.path("prompts.jsonl")
    with failure_arm.open("w") as f:
        for pid in failed:
            f.write(json.dumps(prompts[pid]) + "\n")

    control_dir = Path(rcfg["control_dir"])
    control_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.distill["seed"])
    rolled_ids = [r["id"] for r in rolls]
    control = rng.sample(rolled_ids, k=min(len(failed), len(rolled_ids)))
    with (control_dir / "prompts.jsonl").open("w") as f:
        for pid in control:
            f.write(json.dumps(prompts[pid]) + "\n")

    print(f"[rollout] failure arm: {len(failed)} prompts -> {failure_arm}")
    print(f"[rollout] control arm: {len(control)} random prompts -> {control_dir / 'prompts.jsonl'}")
    return failure_arm
