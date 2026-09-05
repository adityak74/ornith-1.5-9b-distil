"""Where is the pipeline? One screen, no guessing."""

from __future__ import annotations

import json
from pathlib import Path

from .config import Config


def _count(p: Path) -> int:
    return sum(1 for _ in p.open()) if p.exists() else 0


def run(cfg: Config) -> int:
    out = cfg.out_dir
    print(f"run: {out}\n")

    pool = _count(out / "prompts.jsonl")
    wanted = {d: sum(s["n"] for s in srcs) for d, srcs in cfg.distill["prompts"].items()}
    print(f"1 prompts    {pool} built ({', '.join(f'{k} {v}' for k, v in wanted.items())})")

    print("2 teach")
    for name in cfg.models["teachers"]:
        f = out / "teacher" / f"{name}.jsonl"
        n = _count(f)
        if not n:
            print(f"    {name:<11} not started")
            continue
        recs = [json.loads(line) for line in f.open()]
        secs = sum(r["gen_seconds"] for r in recs)
        domains = {d for d in cfg.distill["prompts"] if cfg.teacher_for(d)["name"] == name}
        target = sum(wanted[d] for d in domains)
        rate = secs / max(n, 1)
        eta = (target - n) * rate / 3600
        print(f"    {name:<11} {n}/{target}  {rate:.1f}s/trace  eta {eta:.1f}h")

    stats_f = out / "train" / "stats.json"
    if stats_f.exists():
        st = json.loads(stats_f.read_text())
        keep = st.get("kept", 0) / max(st.get("seen", 1), 1)
        rejects = {k: v for k, v in st.items() if k.startswith("reject")}
        print(f"3 dataset    {st.get('kept')}/{st.get('seen')} kept ({keep:.0%})  {rejects}")
        print(f"             train {_count(out / 'train' / 'train.jsonl')} "
              f"valid {_count(out / 'train' / 'valid.jsonl')}")
    else:
        print("3 dataset    not built")

    adapters = out / "adapters" / "adapters.safetensors"
    print(f"4 train      {'adapters present' if adapters.exists() else 'not started'}")
    for stage, p in (("5 fuse", out / "fused"), ("6 quantize", out / "quant")):
        if p.exists():
            kids = [d.name for d in p.iterdir() if d.is_dir()] if stage.endswith("quantize") else ["fused"]
            print(f"{stage:<12} {', '.join(kids) or 'empty'}")
        else:
            print(f"{stage:<12} not started")

    evals = sorted((out / "eval").glob("*/summary.json"))
    print(f"7 eval       {len(evals)} run(s)")
    for s in evals:
        data = json.loads(s.read_text())
        cells = "  ".join(
            f"{r['benchmark']}={r['accuracy']:.1%}({r['total']})" for r in data["results"]
        )
        print(f"    {s.parent.name:<24} {cells}")
    return 0
