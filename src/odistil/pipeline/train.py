"""Stage 4/5 -- LoRA distillation over the bf16 student, then fuse.

Full-parameter FT of a 9B in bf16 needs roughly 18 GB of weights plus ~72 GB of
fp32 AdamW state; that does not fit in 64 GB unified memory. High-rank LoRA on
every layer over the *full-precision* base gets most of the way there and fuses
back to a plain bf16 checkpoint, which is what we quantize.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from ..config import Config


def _yaml(cfg: Config, adapter_dir: Path) -> Path:
    t = cfg.distill["train"]
    data = cfg.path("train")
    conf = {
        "model": cfg.student_base(),
        "train": True,
        "data": str(data),
        "adapter_path": str(adapter_dir),
        "fine_tune_type": t["fine_tune_type"],
        "num_layers": t["num_layers"],
        "batch_size": t["batch_size"],
        "iters": t["iters"],
        "learning_rate": t["learning_rate"],
        "max_seq_length": t["max_seq_length"],
        "grad_checkpoint": t["grad_checkpoint"],
        "steps_per_eval": t["steps_per_eval"],
        "save_every": t["save_every"],
        "mask_prompt": t["mask_prompt"],
        "seed": cfg.distill["seed"],
        "lr_schedule": {
            "name": t["lr_schedule"],
            "warmup": t["warmup"],
            "arguments": [t["learning_rate"], t["iters"], t["learning_rate"] / 10],
        },
        "lora_parameters": {
            # no explicit "keys": let mlx-lm pick the per-architecture default
            "rank": t["rank"],
            "scale": t["scale"],
            "dropout": t["dropout"],
        },
    }
    import yaml

    p = cfg.path("train_config.yaml")
    p.write_text(yaml.safe_dump(conf, sort_keys=False))
    return p


def train(cfg: Config, resume: bool = False, extra: list[str] | None = None) -> Path:
    adapter_dir = cfg.path("adapters")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    conf = _yaml(cfg, adapter_dir)
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "-c", str(conf)]
    if resume:
        cmd += ["--resume-adapter-file", str(adapter_dir / "adapters.safetensors")]
    cmd += extra or []
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return adapter_dir


def fuse(cfg: Config, out: Path | None = None) -> Path:
    out = out or cfg.path("fused")
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", cfg.student_base(),
        "--adapter-path", str(cfg.path("adapters")),
        "--save-path", str(out),
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)
    (out / "odistil.json").write_text(
        json.dumps({"base": cfg.student_base(), "run": str(cfg.out_dir)}, indent=2)
    )
    return out


def oq_predicate(reference: Path) -> tuple[dict, Callable]:
    """Rebuild an oQ mixed-bit map from an existing oQ checkpoint's config.

    The oQ recipe is affine quantization at a base width with 120 individual
    modules promoted above it -- in oQ4: 110 at 5 bits, 7 at 6 and 3 at 8,
    concentrated in the `linear_attn` projections (out_proj, in_proj_{a,b,z})
    with some `mlp.down_proj` and `self_attn` heads. Reusing the exact map is
    what makes a distilled checkpoint comparable to the shipped oQ4.
    """
    conf = json.loads((reference / "config.json").read_text())
    q = conf.get("quantization") or {}
    base = {k: v for k, v in q.items() if not isinstance(v, dict)}
    per = {k: v for k, v in q.items() if isinstance(v, dict)}
    stripped = {k.removeprefix("language_model."): v for k, v in per.items()}
    hits = {"n": 0}

    def predicate(path: str, module, config) -> bool | dict:
        spec = per.get(path) or stripped.get(path.removeprefix("language_model."))
        if spec:
            hits["n"] += 1
            return dict(spec)
        return True

    predicate.hits = hits
    predicate.per_module = per
    return base, predicate


def quantize(cfg: Config, variant: str, src: Path | None = None) -> Path:
    src = src or cfg.path("fused")
    qcfg = cfg.distill["quantize"]
    recipe = (qcfg.get("recipes") or {}).get(variant)
    dst = cfg.path("quant", variant)

    if recipe:  # external command, if one is configured
        cmd = recipe.format(src=str(src), dst=str(dst))
        print("+", cmd)
        subprocess.run(cmd, shell=True, check=True)
        return dst

    # oQ variants: mirror the mixed-bit map of the shipped checkpoint.
    ref = (cfg.models["student"].get("quantized") or {}).get(variant)
    if variant.startswith("oq") and ref and Path(ref).exists():
        from mlx_lm.convert import convert

        base, predicate = oq_predicate(Path(ref))
        print(f"oQ map from {ref}: base {base['bits']}-bit g{base['group_size']}, "
              f"{len(predicate.per_module)} promoted modules")
        convert(
            hf_path=str(src),
            mlx_path=str(dst),
            quantize=True,
            q_bits=base["bits"],
            q_group_size=base["group_size"],
            q_mode=base.get("mode", "affine"),
            quant_predicate=predicate,
        )
        n, want = predicate.hits["n"], len(predicate.per_module)
        print(f"applied {n}/{want} promoted modules")
        if n == 0 and want:
            print("WARNING: no module names matched -- the fused model's tree differs "
                  "from the reference; inspect the keys before trusting this checkpoint")
        return dst

    spec = next((v for v in qcfg["variants"] if v["name"] == variant), None)
    if not spec:
        raise SystemExit(
            f"unknown quantize variant {variant!r}: not in quantize.variants, no recipe "
            "configured, and no oQ reference checkpoint in models.yaml"
        )
    cmd = [
        sys.executable, "-m", "mlx_lm", "convert",
        "--hf-path", str(src),
        "--mlx-path", str(dst),
        "-q", "--q-bits", str(spec["bits"]), "--q-group-size", str(spec["group_size"]),
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return dst
