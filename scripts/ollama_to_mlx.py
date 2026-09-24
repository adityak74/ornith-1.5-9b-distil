"""Repack an Ollama MLX model into a standard mlx-lm checkpoint, byte for byte.

    uv run python scripts/ollama_to_mlx.py adityakarnam/Ornith-1.5-9B-MLX-distil:latest OUT_DIR

Ollama stores each tensor as its own small safetensors blob: `X.weight` (packed
uint32), `X.weight.scale` and `X.weight.bias` (bf16), with `quant_type` (int4 |
int8) and `group_size` in the blob's metadata. That is MLX affine quantization
under different names, so the repack only renames tensors and writes the
per-module bit widths into config.json -- no weight is re-quantized, and what
oMLX loads is exactly what `ollama pull` downloads.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import mlx.core as mx

MODELS = Path(os.environ.get("OLLAMA_MODELS", "/Volumes/SATECHI/ollama"))


def main(ref: str, out: Path) -> None:
    name, _, tag = ref.partition(":")
    ns, _, model = name.rpartition("/")
    manifest = json.loads((MODELS / "manifests/registry.ollama.ai" / ns / model / (tag or "latest")).read_text())
    blob = lambda d: MODELS / "blobs" / d.replace(":", "-")  # noqa: E731

    out.mkdir(parents=True, exist_ok=True)
    weights: dict[str, mx.array] = {}
    per_module: dict[str, dict] = {}
    kinds: dict[str, int] = {}
    for layer in manifest["layers"]:
        mt, lname = layer["mediaType"], layer.get("name")
        if mt.endswith(".json"):
            (out / lname).write_bytes(blob(layer["digest"]).read_bytes())
            continue
        if not mt.endswith(".tensor"):
            continue
        path = blob(layer["digest"])
        with path.open("rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            meta = json.loads(f.read(n)).get("__metadata__", {})
        tensors = mx.load(str(path), format="safetensors")
        qt = meta.get("quant_type")
        kinds[qt or "unquantized"] = kinds.get(qt or "unquantized", 0) + 1
        if not qt:
            weights.update(tensors)
            continue
        module = lname.removesuffix(".weight")
        weights[f"{module}.weight"] = tensors[lname]
        weights[f"{module}.scales"] = tensors[f"{lname}.scale"]
        weights[f"{module}.biases"] = tensors[f"{lname}.bias"]
        bits = {"int4": 4, "int8": 8}[qt]
        if bits != 4:
            per_module[module] = {"group_size": int(meta["group_size"]), "bits": bits}

    conf = json.loads((out / "config.json").read_text())
    q = {"group_size": 64, "bits": 4, "mode": "affine", **per_module}
    conf["quantization"] = conf["quantization_config"] = q
    (out / "config.json").write_text(json.dumps(conf, indent=2))
    mx.save_safetensors(str(out / "model.safetensors"), weights, metadata={"format": "mlx"})
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum(v.nbytes for v in weights.values())},
         "weight_map": {k: "model.safetensors" for k in sorted(weights)}}, indent=2))
    print(f"{ref}: {kinds} -> {len(weights)} tensors, {len(per_module)} int8 modules, {out}")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
