"""Teacher top-k logprobs for logit-level distillation.

Every run so far trained on one sampled completion per prompt: the student sees
which token the teacher happened to emit, and nothing about how confident it
was or what it nearly said instead. Logit distillation trains against the
teacher's full next-token distribution, which the shared 248,044-token
vocabulary makes possible.

The traces already exist, so the teacher does not need to generate again --
teacher-forcing the kept sequence gives every distribution in one forward pass.
Measured on a 35B-A3B teacher: 1.3 s for a 1,000-token sample, against the
~6 s it took to sample that trace in the first place.

Only the top k are stored. At k=64 that captures 99.9% of the probability mass
on this model, so the sparse approximation costs almost nothing, and the
storage is ~0.3 MB per sample instead of ~1 GB dense.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from .config import Config

TOP_K = 64


def _domain_index(cfg: Config) -> dict[str, str]:
    """Map a prompt to the domain that produced it, so each sample is scored
    by the teacher that wrote it."""
    index: dict[str, str] = {}
    for path in sorted((cfg.out_dir / "teacher").glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                rec = json.loads(line)
                index[rec["prompt"]] = rec["domain"]
    return index


def _sequence(tok, messages: list[dict]) -> tuple[list[int], int]:
    """Full token sequence and the index where the assistant's turn starts."""
    prompt = tok.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False)
    full = prompt + messages[-1]["content"]
    return tok.encode(full), len(tok.encode(prompt))


def extract(cfg: Config, split: str = "train", top_k: int = TOP_K, limit: int | None = None) -> Path:
    """Write teacher top-k logprobs for every sample in `split`.

    Output is one `.safetensors` per teacher holding ragged arrays plus an
    offsets table, keyed by the line number of the sample in the split file.
    """
    from mlx_lm import load

    from .mlxutil import load_tokenizer

    src = cfg.path("train", f"{split}.jsonl")
    rows = [json.loads(line) for line in src.open()]
    if limit:
        rows = rows[:limit]
    domains = _domain_index(cfg)

    by_teacher: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        prompt = row["messages"][-2]["content"]
        domain = domains.get(prompt)
        if domain is None:
            continue
        by_teacher.setdefault(cfg.teacher_for(domain)["name"], []).append(i)

    out_dir = cfg.path("teacher_logits")
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(cfg.student_base())

    for teacher_name, indices in by_teacher.items():
        teacher = cfg.teacher(teacher_name)
        out_path = out_dir / f"{split}-{teacher_name}.safetensors"
        if out_path.exists():
            print(f"[{teacher_name}] {out_path.name} exists, skipping")
            continue
        print(f"[{teacher_name}] scoring {len(indices)} samples with {teacher['path']}", flush=True)
        model, _ = load(teacher["path"])

        part_dir = out_dir / f"{split}-{teacher_name}.parts"
        part_dir.mkdir(exist_ok=True)

        for n, i in enumerate(indices):
            part = part_dir / f"{i}.safetensors"
            if part.exists():          # resume after an interrupted run
                continue
            token_ids, prompt_len = _sequence(tok, rows[i]["messages"])
            logits = model(mx.array([token_ids]))[0, prompt_len - 1 : -1]
            # top-k by logit is top-k by logprob; subtracting logsumexp after
            # the gather avoids a second [T, 248k] tensor, which is what made
            # this run out of memory.
            lse = mx.logsumexp(logits.astype(mx.float32), axis=-1, keepdims=True)
            idx = mx.argpartition(-logits, top_k, axis=-1)[:, :top_k]
            val = mx.take_along_axis(logits.astype(mx.float32), idx, axis=-1) - lse
            mx.eval(idx, val)
            mx.save_safetensors(
                str(part),
                {"ids": idx.astype(mx.int32), "logprobs": val.astype(mx.float16),
                 "prompt_len": mx.array([prompt_len], dtype=mx.int32)},
            )
            del logits, lse, idx, val
            if (n + 1) % 25 == 0:
                mx.clear_cache()
                print(f"  {n + 1}/{len(indices)}", flush=True)

        # stitch the per-sample parts into one shard
        ids_all, lps_all, offsets, sample_ids, starts = [], [], [0], [], []
        for i in indices:
            blob = mx.load(str(part_dir / f"{i}.safetensors"))
            ids_all.append(blob["ids"])
            lps_all.append(blob["logprobs"])
            offsets.append(offsets[-1] + blob["ids"].shape[0])
            sample_ids.append(i)
            starts.append(int(blob["prompt_len"][0]))

        mx.save_safetensors(
            str(out_path),
            {
                "ids": mx.concatenate(ids_all, axis=0),
                "logprobs": mx.concatenate(lps_all, axis=0),
                "offsets": mx.array(offsets, dtype=mx.int32),
                "sample": mx.array(sample_ids, dtype=mx.int32),
                "prompt_len": mx.array(starts, dtype=mx.int32),
            },
        )
        for f in part_dir.glob("*.safetensors"):
            f.unlink()
        part_dir.rmdir()
        size = out_path.stat().st_size / 1024**2
        print(f"[{teacher_name}] -> {out_path.name} ({size:.0f} MB)", flush=True)
        del model
        mx.clear_cache()

    return out_dir


def load_teacher_logits(cfg: Config, split: str = "train") -> dict[int, tuple[mx.array, mx.array]]:
    """Per-sample (top-k ids, top-k logprobs), keyed by line number."""
    out: dict[int, tuple[mx.array, mx.array]] = {}
    for path in sorted((cfg.out_dir / "teacher_logits").glob(f"{split}-*.safetensors")):
        blob = mx.load(str(path))
        offsets = blob["offsets"].tolist()
        samples = blob["sample"].tolist()
        for n, sample in enumerate(samples):
            lo, hi = offsets[n], offsets[n + 1]
            out[int(sample)] = (blob["ids"][lo:hi], blob["logprobs"][lo:hi])
    return out
