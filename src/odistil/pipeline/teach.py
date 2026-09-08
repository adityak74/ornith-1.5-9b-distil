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


def _style_for(teacher: dict, domain: str) -> str | None:
    """Per-domain style, falling back to the teacher's default.

    Brevity is not uniformly good: shortening the code teacher's reasoning cost
    5.4 pp of HumanEval (DECISIONS.md 29). It is right for abstention, where the
    judgement is "the passage does not say this" and a thousand tokens of
    deliberation only pushes the sample past the training length cap.
    """
    by_domain = teacher.get("style_by_domain") or {}
    return by_domain.get(domain, teacher.get("style"))


def _gen(teacher: dict, tconf: dict, bs: int, items: list[dict], budget: int):
    system = " ".join(filter(None, [SYSTEM, _style_for(teacher, items[0]["domain"])]))
    return generate_batch(
        teacher["path"],
        [c["prompt"] for c in items],
        system=system,
        max_tokens=budget,
        temp=teacher["temp"],
        top_p=teacher["top_p"],
        think=tconf["think"],
        batch_size=bs,
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
        retries = tconf.get("max_retries", 0)
        # Batch within a domain: one system prompt per batch, and styles differ.
        todo.sort(key=lambda r: r["domain"])
        with out.open("a") as f:
            for i in range(0, len(todo), bs):
                chunk = todo[i : i + bs]
                chunk = [c for c in chunk if c["domain"] == chunk[0]["domain"]]

                comps = _gen(t, tconf, bs, chunk, t["max_tokens"])

                # A trace that ran out of budget mid-reasoning has no answer and
                # would be thrown away by the verifier -- the generation time is
                # already spent, so give those items one longer attempt.
                for attempt in range(retries):
                    stuck = [j for j, c in enumerate(comps) if c.truncated]
                    if not stuck:
                        break
                    budget = int(t["max_tokens"] * 1.5 ** (attempt + 1))
                    print(f"  retrying {len(stuck)} truncated at {budget} tokens")
                    redone = _gen(t, tconf, bs, [chunk[j] for j in stuck], budget)
                    for j, c in zip(stuck, redone, strict=True):
                        if not c.truncated:
                            comps[j] = c

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
                                "truncated": comp.truncated,
                            }
                        )
                        + "\n"
                    )
                    f.flush()
                print(f"  {min(i + bs, len(todo))}/{len(todo)}", end="\r", flush=True)
        print(f"\n[{tname}] -> {out}")
        written.append(out)
    return written
