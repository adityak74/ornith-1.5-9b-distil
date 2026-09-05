"""Stage 2 -- teacher generation.

Routes each prompt to the teacher that owns its domain (Qwen3.6-35B for
knowledge/truthfulness, Ornith-1.5-35B for code), generates with thinking on,
and appends to data/teacher/<teacher>.jsonl. Resumable: ids already present in
the output file are skipped.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ..config import Config
from ..mlxutil import generate_batch

SYSTEM = (
    "You are a careful expert. Reason step by step, keep the reasoning tight, "
    "and follow the requested output format exactly."
)


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def run(cfg: Config, teachers: list[str] | None = None, limit: int | None = None) -> list[Path]:
    prompts_file = cfg.path("prompts.jsonl")
    if not prompts_file.exists():
        raise SystemExit("no prompts.jsonl -- run `odistil prompts` first")

    by_teacher: dict[str, list[dict]] = defaultdict(list)
    with prompts_file.open() as f:
        for line in f:
            rec = json.loads(line)
            by_teacher[cfg.teacher_for(rec["domain"])["name"]].append(rec)

    tconf = cfg.distill["teach"]
    written = []
    for tname, recs in by_teacher.items():
        if teachers and tname not in teachers:
            continue
        t = cfg.teacher(tname)
        out = cfg.path("teacher", f"{tname}.jsonl")
        done = _done_ids(out)
        todo = [r for r in recs if r["id"] not in done]
        if limit:
            todo = todo[:limit]
        print(f"[{tname}] {t['path']}: {len(todo)} to generate ({len(done)} cached)")

        bs = tconf["batch_size"]
        with out.open("a") as f:
            for i in range(0, len(todo), bs):
                chunk = todo[i : i + bs]
                comps = generate_batch(
                    t["path"],
                    [c["prompt"] for c in chunk],
                    system=SYSTEM,
                    max_tokens=t["max_tokens"],
                    temp=t["temp"],
                    top_p=t["top_p"],
                    think=tconf["think"],
                    batch_size=bs,
                )
                for rec, comp in zip(chunk, comps, strict=True):
                    f.write(
                        json.dumps(
                            {
                                **rec,
                                "teacher": tname,
                                "teacher_model": t["path"],
                                "answer": comp.text,
                                "think": comp.think,
                                "raw": comp.raw,
                                "gen_tokens": comp.tokens,
                                "gen_seconds": round(comp.seconds, 2),
                            }
                        )
                        + "\n"
                    )
                    f.flush()
                print(f"  {min(i + bs, len(todo))}/{len(todo)}", end="\r", flush=True)
        print(f"\n[{tname}] -> {out}")
        written.append(out)
    return written
