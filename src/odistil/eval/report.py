"""Merge run summaries with benchmarks/baseline.json into one comparison table."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import Config

BENCHES = ["mmlu", "truthfulqa", "humaneval"]


def collect(cfg: Config) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    baseline = json.loads((cfg.root / "benchmarks" / "baseline.json").read_text())
    for name, res in baseline["models"].items():
        rows[name] = {b: res.get(b) for b in BENCHES}
    for summary in sorted((cfg.out_dir / "eval").glob("*/summary.json")):
        data = json.loads(summary.read_text())
        name = f"{summary.parent.name} (this run)"
        rows[name] = {
            r["benchmark"]: {
                "accuracy": r["accuracy"],
                "correct": r["correct"],
                "total": r["total"],
                "seconds": r["seconds"],
            }
            for r in data["results"]
        }
    return rows


def table(cfg: Config, show_time: bool = False) -> str:
    rows = collect(cfg)
    targets = json.loads((cfg.root / "benchmarks" / "baseline.json").read_text())["targets"]
    w = max(len(n) for n in rows) + 2
    head = f"{'Model':<{w}}" + "".join(f"{b.upper():>14}" for b in BENCHES)
    lines = [head, "-" * len(head)]
    for name, res in rows.items():
        cells = []
        for b in BENCHES:
            r = res.get(b)
            if not r:
                cells.append(f"{'-':>14}")
            elif show_time:
                cells.append(f"{r['accuracy']:>8.1%} {r['seconds'] / 3600:>4.1f}h")
            else:
                cells.append(f"{r['accuracy']:>14.1%}")
        lines.append(f"{name:<{w}}" + "".join(cells))
    lo, hi = "target (low)", "target (high)"
    for label, i in ((lo, 0), (hi, 1)):
        cells = "".join(f"{targets['distilled-9B'][b][i]:>14.1%}" for b in BENCHES)
        lines.append(f"{label:<{w}}" + cells)
    return "\n".join(lines)


def write(cfg: Config) -> Path:
    p = cfg.path("report.txt")
    p.write_text(table(cfg) + "\n\n" + table(cfg, show_time=True) + "\n")
    return p
