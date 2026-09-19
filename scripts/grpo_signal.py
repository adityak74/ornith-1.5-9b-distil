"""GRPO signal check: does sampling G completions per prompt give groups with
both a verified-correct and a failed member? Without those there is no
group-relative advantage and no gradient for termination (DECISIONS.md 50).

    uv run python scripts/grpo_signal.py [--prompts 100] [--group 8] [--temp 0.8]

Writes runs/grpo-signal/samples.jsonl (resumable) and prints the group stats.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from odistil.config import Config
from odistil.mlxutil import generate_batch
from odistil.pipeline.dataset import _verify
from odistil.pipeline.teach import SYSTEM

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", type=int, default=100)
ap.add_argument("--group", type=int, default=8)
ap.add_argument("--temp", type=float, default=0.8)
ap.add_argument("--max-tokens", type=int, default=2048)
ap.add_argument("--kind", default="code")
ap.add_argument("--student", default="/Volumes/SATECHI/.omlx/Ornith-1.5-9B-MLX-distil-oQ4")
a = ap.parse_args()

cfg = Config.load(None, "configs/v8.yaml")
pool = [json.loads(l) for l in open("runs/v8pool/prompts.jsonl") if l.strip()]
pool = [p for p in pool if p["kind"] == a.kind]
random.Random(50).shuffle(pool)
pool = pool[: a.prompts]
out = Path("runs/grpo-signal"); out.mkdir(exist_ok=True)
samples = out / "samples.jsonl"
done = Counter(json.loads(l)["id"] for l in samples.open()) if samples.exists() else Counter()

with samples.open("a") as f:
    for i, p in enumerate(pool):
        if done[p["id"]] >= a.group:
            continue
        comps = generate_batch(a.student, [p["prompt"]] * a.group, system=SYSTEM,
                               max_tokens=a.max_tokens, temp=a.temp, think=True, batch_size=a.group)
        for c in comps:
            ok, why = _verify({**p, "answer": c.text}, cfg.distill["dataset"])
            f.write(json.dumps({"id": p["id"], "correct": bool(ok), "truncated": c.truncated,
                                "tokens": c.tokens}) + "\n")
        f.flush()
        print(f"  {i + 1}/{len(pool)}", end="\r", flush=True)

groups: dict[str, list[dict]] = {}
for l in samples.open():
    r = json.loads(l); groups.setdefault(r["id"], []).append(r)
n = len(groups)
mixed = sum(1 for g in groups.values() if any(r["correct"] for r in g) and not all(r["correct"] for r in g))
mixed_trunc = sum(1 for g in groups.values() if any(r["correct"] for r in g) and any(r["truncated"] for r in g))
all_fail = sum(1 for g in groups.values() if not any(r["correct"] for r in g))
all_pass = sum(1 for g in groups.values() if all(r["correct"] for r in g))
tot = [r for g in groups.values() for r in g]
print(f"\n[grpo-signal] {n} prompts x {a.group} samples, temp {a.temp}, {a.max_tokens} tokens")
print(f"  sample accuracy {sum(r['correct'] for r in tot) / len(tot):.1%}, truncated {sum(r['truncated'] for r in tot) / len(tot):.1%}")
print(f"  groups: mixed {mixed} ({mixed / n:.0%})   mixed with a truncated member {mixed_trunc} ({mixed_trunc / n:.0%})"
      f"   all-fail {all_fail}   all-pass {all_pass}")
