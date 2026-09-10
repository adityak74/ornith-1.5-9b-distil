"""Config loading and run-directory helpers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"


_ENV_REF = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def _expand(value: Any) -> Any:
    """Resolve ${VAR} and ${VAR:-default} in config strings.

    Model locations differ per machine, so paths are written against
    OMLX_MODEL_DIR with the author's oMLX model_dir as the default.
    """
    if isinstance(value, str):
        return _ENV_REF.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return _expand(yaml.safe_load(f))


@dataclass
class Config:
    models: dict[str, Any]
    distill: dict[str, Any]
    root: Path = ROOT

    @classmethod
    def load(cls, models: str | Path | None = None, distill: str | Path | None = None) -> Config:
        return cls(
            models=_load(Path(models or CONFIGS / "models.yaml")),
            distill=_load(Path(distill or CONFIGS / "distill.yaml")),
        )

    # -- paths ---------------------------------------------------------------
    @property
    def out_dir(self) -> Path:
        p = self.root / self.distill["out_dir"]
        p.mkdir(parents=True, exist_ok=True)
        return p

    def path(self, *parts: str) -> Path:
        p = self.out_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # -- models --------------------------------------------------------------
    def student_base(self) -> str:
        s = self.models["student"]["base"]
        return s["local"] or s["repo"]

    def train_base(self) -> str:
        """Model the trainer attaches adapters to.

        Defaults to the full-precision student. Set `train.base` to a quantized
        checkpoint (or a `student:<variant>` ref) for QLoRA, where the adapter
        learns against the quantized forward pass instead of a clean one.
        """
        ref = self.distill["train"].get("base")
        return self.model_ref(ref) if ref else self.student_base()

    def teacher(self, name: str) -> dict[str, Any]:
        t = dict(self.models["teachers"][name])
        t["path"] = t.get("local") or t["repo"]
        t["name"] = name
        return t

    def teacher_for(self, domain: str) -> dict[str, Any]:
        for name in self.models["teachers"]:
            t = self.teacher(name)
            if domain in t["domains"]:
                return t
        raise KeyError(f"no teacher declares domain {domain!r}")

    def model_ref(self, ref: str) -> str:
        """Resolve 'student', 'student:oq4', 'teacher:code', a repo id, or a path."""
        if ref == "student":
            return self.student_base()
        if ref.startswith("student:"):
            return self.models["student"]["quantized"][ref.split(":", 1)[1]]
        if ref.startswith("teacher:"):
            return self.teacher(ref.split(":", 1)[1])["path"]
        return ref


def hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
