"""Runs tasks against a model and writes per-item JSONL + a summary JSON.

Resumable: an interrupted run picks up from the items already in the JSONL.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..config import Config
from ..mlxutil import generate_batch
from .tasks import REGISTRY, Task

SYSTEM = "You are a careful expert. Follow the requested output format exactly."


def _load_done(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open() as f:
        return {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}


def run_task(
    model_path: str,
    task: Task,
    out_dir: Path,
    *,
    think: bool = True,
    max_tokens: int = 2048,
    temp: float = 0.0,
    batch_size: int = 8,
    limit: int | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = out_dir / f"{task.name}.jsonl"
    done = _load_done(detail)
    items = task.items[:limit] if limit else task.items
    todo = [it for it in items if it.id not in done]
    print(f"[{task.name}] {len(todo)} to run ({len(done)} cached) on {model_path}")

    t0 = time.time()
    with detail.open("a") as f:
        for i in range(0, len(todo), batch_size):
            chunk = todo[i : i + batch_size]
            comps = generate_batch(
                model_path,
                [it.prompt for it in chunk],
                system=SYSTEM,
                max_tokens=max_tokens,
                temp=temp,
                think=think,
                batch_size=batch_size,
            )
            for it, comp in zip(chunk, comps, strict=True):
                rec = {
                    "id": it.id,
                    "correct": bool(task.score(it, comp.text)),
                    "output": comp.text,
                    "truncated": comp.truncated,
                    "think_chars": len(comp.think or ""),
                    "tokens": comp.tokens,
                    "seconds": round(comp.seconds, 2),
                    "meta": it.meta,
                }
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done[it.id] = rec
            n = min(i + batch_size, len(todo))
            acc = sum(r["correct"] for r in done.values()) / max(len(done), 1)
            print(f"  {n}/{len(todo)}  running acc {acc:.1%}", end="\r", flush=True)

    recs = [done[it.id] for it in items if it.id in done]
    correct = sum(r["correct"] for r in recs)
    summary = {
        "benchmark": task.name,
        "model": model_path,
        "total": len(recs),
        "correct": correct,
        "accuracy": round(correct / max(len(recs), 1), 4),
        "truncated": sum(r.get("truncated", False) for r in recs),
        "seconds": round(sum(r["seconds"] for r in recs) or time.time() - t0, 1),
        "think": think,
    }
    trunc = summary["truncated"]
    note = f"  [{trunc} truncated before answering -- raise eval.max_tokens]" if trunc else ""
    print(f"\n[{task.name}] {summary['accuracy']:.1%} ({correct}/{len(recs)}){note}")
    return summary


def run(
    cfg: Config,
    model_ref: str,
    benchmarks: list[str] | None = None,
    limit: int | None = None,
    tag: str | None = None,
) -> Path:
    ecfg = cfg.distill["eval"]
    model_path = cfg.model_ref(model_ref)
    tag = tag or model_ref.replace("/", "_").replace(":", "-")
    out_dir = cfg.path("eval", tag)

    summaries = []
    for name in benchmarks or ecfg["benchmarks"]:
        builder = REGISTRY[name]
        task = builder(sample=ecfg["mmlu_sample"], seed=cfg.distill["seed"]) if name == "mmlu" else builder()
        summaries.append(
            run_task(
                model_path,
                task,
                out_dir,
                think=ecfg["think"],
                max_tokens=ecfg["max_tokens"],
                temp=ecfg["temp"],
                limit=limit,
            )
        )

    # Merge with whatever this tag already measured: running one benchmark at a
    # time must not wipe the others.
    path = out_dir / "summary.json"
    merged = {}
    if path.exists():
        for r in json.loads(path.read_text()).get("results", []):
            merged[r["benchmark"]] = r
    for r in summaries:
        merged[r["benchmark"]] = r
    path.write_text(
        json.dumps({"model_ref": model_ref, "results": list(merged.values())}, indent=2)
    )
    print(f"-> {path}")
    return path
