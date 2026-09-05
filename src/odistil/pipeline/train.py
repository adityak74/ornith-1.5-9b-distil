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


def quantize(cfg: Config, variant: str, src: Path | None = None) -> Path:
    src = src or cfg.path("fused")
    qcfg = cfg.distill["quantize"]
    recipe = (qcfg.get("recipes") or {}).get(variant)
    dst = cfg.path("quant", variant)

    if recipe:  # external oQ recipe
        cmd = recipe.format(src=str(src), dst=str(dst))
        print("+", cmd)
        subprocess.run(cmd, shell=True, check=True)
        return dst

    spec = next((v for v in qcfg["variants"] if v["name"] == variant), None)
    if not spec:
        raise SystemExit(f"unknown quantize variant {variant!r}; no recipe configured either")
    cmd = [
        sys.executable, "-m", "mlx_lm", "convert",
        "--hf-path", str(src),
        "--mlx-path", str(dst),
        "-q", "--q-bits", str(spec["bits"]), "--q-group-size", str(spec["group_size"]),
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return dst
