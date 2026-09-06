"""Final stage -- turn a checkpoint directory into something oMLX can load.

`mlx_lm fuse`/`convert` write weights and usually the tokenizer, but a model
dir is only loadable by the oMLX server if it looks exactly like the shipped
ones: config.json, the safetensors shards plus their index, tokenizer.json,
tokenizer_config.json and chat_template.jinja. This stage fills in whatever is
missing from the base model, writes a model card, and proves the result loads
and generates before calling it done.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..config import Config

REQUIRED = ["config.json", "tokenizer.json", "tokenizer_config.json"]
COPY_IF_MISSING = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]


def _size_gb(d: Path) -> float:
    return sum(f.stat().st_size for f in d.glob("*.safetensors")) / 1024**3


def package(cfg: Config, src: Path, name: str, install: bool = False) -> Path:
    src = Path(src)
    if not src.exists():
        raise SystemExit(f"{src} does not exist")
    base = Path(cfg.student_base())

    for fname in COPY_IF_MISSING:
        target = src / fname
        if not target.exists() and (base / fname).exists():
            shutil.copy2(base / fname, target)
            print(f"copied {fname} from the base model")

    missing = [f for f in REQUIRED if not (src / f).exists()]
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        missing.append("*.safetensors")
    if len(shards) > 1 and not (src / "model.safetensors.index.json").exists():
        missing.append("model.safetensors.index.json")
    if missing:
        raise SystemExit(f"{src} is not loadable, missing: {', '.join(missing)}")

    quant = (json.loads((src / "config.json").read_text()).get("quantization") or {})
    bits = quant.get("bits", "bf16 (unquantized)")
    card = src / "MODEL_CARD.md"
    card.write_text(_card(cfg, name, src, bits))

    print(f"{name}: {len(shards)} shard(s), {_size_gb(src):.1f} GB, quantization: {bits}")
    print("verifying it loads and generates...")
    from ..mlxutil import generate_one

    out = generate_one(str(src), "Write a one-line Python function that returns the square of x.",
                       max_tokens=256, think=False)
    ok = "def" in out.raw or "lambda" in out.raw
    print(f"  {'ok' if ok else 'SUSPECT'}: {out.raw.strip()[:120]!r}")
    if not ok:
        print("  WARNING: the model loaded but its output does not look like code")

    if install:
        dest = Path(cfg.models["install_dir"]) / name
        if dest.exists():
            raise SystemExit(f"{dest} already exists -- remove it first")
        shutil.copytree(src, dest)
        print(f"installed -> {dest}")
        return dest

    print("\nto use in oMLX, copy it into the server's model dir:")
    print(f"  cp -R {src} {cfg.models['install_dir']}/{name}")
    return src


def _results_table(cfg: Config) -> str:
    """Measured results, if this run has any, as a markdown table."""
    rows: dict[str, dict] = {}
    for summary in sorted((cfg.out_dir / "eval").glob("*/summary.json")):
        results = json.loads(summary.read_text())["results"]
        # Skip sanity checks and superseded runs: a handful of items is not a
        # measurement, and a stale token budget is not comparable.
        if max((r["total"] for r in results), default=0) < 100:
            continue
        if "budget" in summary.parent.name or "sanity" in summary.parent.name:
            continue
        for r in results:
            rows.setdefault(r["benchmark"], {})[summary.parent.name] = r
    if not rows:
        return "_Not yet measured._"
    tags = sorted({t for v in rows.values() for t in v})
    out = ["| benchmark | " + " | ".join(tags) + " |",
           "|---|" + "---|" * len(tags)]
    for bench, by_tag in rows.items():
        cells = []
        for t in tags:
            r = by_tag.get(t)
            cells.append(f"{r['accuracy']:.1%} (n={r['total']})" if r else "-")
        out.append(f"| {bench} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def _card(cfg: Config, name: str, src: Path, bits) -> str:
    run = cfg.out_dir
    stats_file = run / "train" / "stats.json"
    stats = json.loads(stats_file.read_text()) if stats_file.exists() else {}
    results = _results_table(cfg)
    return f"""# {name}

Distilled variant of `Ornith-1.5-9B-MLX`, in MLX format.

- **Base:** `{cfg.student_base()}` (bf16)
- **Teachers:** `{cfg.teacher('knowledge')['repo']}` for knowledge and
  truthfulness, `{cfg.teacher('code')['repo']}` for code
- **Method:** sequence-level distillation on rejection-sampled teacher traces,
  rank-{cfg.distill['train']['rank']} LoRA over the bf16 base, fused, then
  quantized
- **Quantization:** {bits}
- **Size:** {_size_gb(src):.1f} GB
- **Training data:** {stats.get('kept', '?')} verified traces kept of
  {stats.get('seen', '?')} generated

## Measured

All numbers below are from this repo's `odistil eval` harness (greedy, thinking
on, 4096-token budget for HumanEval and 3072 for MMLU). They are **not**
comparable to numbers from the oMLX server, which uses different prompting,
parsing and token budgets.

{results}

Every training sample was verified before use: multiple-choice answers against
the gold letter, open questions against gold aliases, and code by executing the
source dataset's own tests. Prompts sharing a 13-gram with MMLU test,
TruthfulQA or HumanEval were dropped.

Benchmarks and method: see `DECISIONS.md` and `docs/PLAN.md` in the
`ornith-1.5-9b-distil` repo. Numbers measured by `odistil eval` are not
comparable to numbers measured by the oMLX server -- compare like with like.
"""
