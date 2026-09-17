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

`rollout.mode: harvest` (v9) inverts the selection: the student decodes under
HALT (`force_budget`) and every *verified-correct* trace becomes a training
row in teacher/self.jsonl, tagged `forced` when the harness closed the
reasoning for it. That is self-distillation of the harness fix: the traces
that end early and still verify are exactly the behaviour quantization lost.
`harvest.forced: false` drops the forced rows, which is the control arm.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from ..config import Config
from ..mlxutil import FORCE_SENTENCE, generate_batch
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
    # group by kind so one budget applies per batch; batch WITHIN each group
    # (filtering a mixed chunk to its first kind and advancing by bs silently
    # drops the rest of that chunk at every boundary)
    groups: dict[str, list[dict]] = {}
    for p in todo:
        groups.setdefault(p["kind"], []).append(p)
    batches = [g[i : i + bs] for g in groups.values() for i in range(0, len(g), bs)]
    n_done = 0
    with out.open("a") as f:
        for chunk in batches:
            comps = generate_batch(
                student,
                [c["prompt"] for c in chunk],
                system=SYSTEM,
                max_tokens=_budget(rcfg, chunk[0]["kind"]),
                temp=0.0,
                think=True,
                batch_size=bs,
                force_budget=rcfg.get("force_budget"),
            )
            for rec, comp in zip(chunk, comps, strict=True):
                ok, why = _verify({**rec, "answer": comp.text}, dcfg)
                f.write(json.dumps({
                    "id": rec["id"],
                    "domain": rec["domain"],
                    "kind": rec["kind"],
                    "correct": bool(ok),
                    "truncated": comp.truncated,
                    "forced": comp.forced,
                    "tokens": comp.tokens,
                    "why": why,
                    # kept so a verifier fix can re-score without regenerating,
                    # and so `harvest` can turn a correct roll into a training row
                    "answer": comp.text,
                    "think": comp.think,
                }) + "\n")
                f.flush()
            n_done += len(chunk)
            print(f"  {n_done}/{len(todo)}", end="\r", flush=True)

    return harvest(cfg) if rcfg.get("mode") == "harvest" else triage(cfg)


def _load_rollouts(cfg: Config) -> list[dict]:
    path = Path(cfg.distill["rollout"].get("rollouts") or cfg.path("rollouts.jsonl"))
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def strip_force(think: str | None) -> str | None:
    """Drop the harness's budget sentence so the model learns to *stop*, not to
    recite that it was stopped. The </think> that followed it stays, because
    `_target` re-adds it: the training row closes the block right where the
    harness closed it."""
    if not think:
        return think
    think = think.rstrip()
    if think.endswith(FORCE_SENTENCE):
        think = think[: -len(FORCE_SENTENCE)].rstrip()
    return think


def harvest(cfg: Config) -> Path:
    """Verified-correct rollouts -> teacher/self.jsonl, in the teacher record
    shape so `dataset` re-verifies and packs them like any other trace."""
    rcfg = cfg.distill["rollout"]
    hcfg = rcfg.get("harvest") or {}
    keep_forced = hcfg.get("forced", True)
    with Path(rcfg["pool"]).open() as f:
        prompts = {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}
    rolls = _load_rollouts(cfg)

    out = cfg.path("teacher", "self.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict[str, int]] = {}
    n_kept = 0
    with out.open("w") as f:
        for r in rolls:
            k = stats.setdefault(r["kind"], {"n": 0, "natural": 0, "forced": 0, "dropped_forced": 0})
            k["n"] += 1
            if not r["correct"]:
                continue
            if r.get("forced"):
                if not keep_forced:
                    k["dropped_forced"] += 1
                    continue
                k["forced"] += 1
            else:
                k["natural"] += 1
            src = prompts[r["id"]]
            f.write(json.dumps({
                **src,
                "think": strip_force(r.get("think")),
                "answer": r["answer"],
                "forced": bool(r.get("forced")),
                "source": "self",
            }) + "\n")
            n_kept += 1

    print("\n[harvest] verified-correct rollouts kept as training rows:")
    for kind, k in sorted(stats.items()):
        print(f"  {kind:<10} natural {k['natural']:4d}   forced {k['forced']:4d}   "
              f"dropped(forced) {k['dropped_forced']:4d}   (n={k['n']})")
    print(f"[harvest] {n_kept} rows -> {out}")

    # the other half of the mix: the teacher traces the base run learned from,
    # copied so `dataset` sees one teacher/ directory and one provenance
    for extra in hcfg.get("include_teacher") or []:
        dst = out.parent / Path(extra).name
        shutil.copyfile(extra, dst)
        print(f"[harvest] included {extra} -> {dst}")
    return out


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
