"""Preflight: is this machine actually able to run the pipeline?"""

from __future__ import annotations

import shutil
from pathlib import Path

from .config import Config

GB = 1024**3


def _local(path: str) -> bool:
    return Path(path).expanduser().exists()


def _cached(repo: str) -> bool:
    from huggingface_hub import scan_cache_dir

    try:
        return any(r.repo_id == repo for r in scan_cache_dir().repos if r.size_on_disk > GB)
    except Exception:  # noqa: BLE001 -- cache scan is best-effort
        return False


def _all_paths(cfg: Config) -> list[str]:
    paths = [cfg.student_base()]
    paths += [cfg.teacher(n)["path"] for n in cfg.models["teachers"]]
    paths += list(cfg.models["student"].get("quantized", {}).values())
    return paths


def run(cfg: Config, download: bool = False) -> int:
    import mlx.core as mx

    print(f"mlx {getattr(mx, '__version__', '?')}   metal={mx.metal.is_available()}")
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    mem = info.get("max_recommended_working_set_size", 0)
    print(f"recommended working set: {mem / GB:.0f} GB")
    free = shutil.disk_usage(Path.home()).free / GB
    print(f"free disk (internal, holds runs/): {free:.0f} GB  "
          "(need ~40 GB for the fused bf16 checkpoint + quants)")

    # Weights live on an external volume; a missing mount looks exactly like a
    # missing model otherwise.
    vols = {Path(p).parents[-4] for p in _all_paths(cfg) if str(p).startswith("/Volumes/")}
    for v in sorted(vols):
        ok = v.exists()
        print(f"volume {v}: {'mounted, ' + f'{shutil.disk_usage(v).free / GB:.0f} GB free' if ok else 'NOT MOUNTED'}")

    refs = [("student", cfg.student_base())] + [
        (f"teacher:{n}", cfg.teacher(n)["path"]) for n in cfg.models["teachers"]
    ]
    missing = []
    print("\nmodels:")
    for label, path in refs:
        have = _local(path) or _cached(path)
        print(f"  {'ok ' if have else 'MISS'}  {label:<16} {path}")
        if not have:
            missing.append(path)

    if missing and download:
        from huggingface_hub import snapshot_download

        for repo in missing:
            print(f"downloading {repo} ...")
            snapshot_download(repo)
        missing = []

    if missing:
        print("\nmissing models -- run `odistil check --download` or set `local:` paths "
              "in configs/models.yaml")
        return 1

    print("\nquantized student variants (eval targets / oQ references):")
    for name, path in (cfg.models["student"].get("quantized") or {}).items():
        print(f"  {'ok ' if _local(path) or _cached(path) else 'MISS'}  {name:<16} {path}")

    print("\ntokenizer compatibility (decides whether logit distillation is possible):")
    from .mlxutil import tokenizer_fingerprint

    student = tokenizer_fingerprint(cfg.student_base())
    for name in cfg.models["teachers"]:
        t = tokenizer_fingerprint(cfg.teacher(name)["path"])
        same = t["probe_ids"] == student["probe_ids"] and t["vocab_size"] == student["vocab_size"]
        verdict = "shared vocab -> logit distillation viable" if same else "different vocab -> sequence-level only"
        print(f"  teacher:{name:<10} vocab={t['vocab_size']:<7} {verdict}")
    print(f"  student{'':<11} vocab={student['vocab_size']}")
    return 0
