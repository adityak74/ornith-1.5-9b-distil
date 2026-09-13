"""Logit-level distillation: train against the teacher's distribution.

Every earlier run minimised cross-entropy against a single sampled completion.
The teacher's distribution carries strictly more: how confident it was at each
step, and what it nearly said instead. Here the loss is

    L = alpha * CE(student, sampled token) + (1 - alpha) * KL(teacher || student)

with the KL taken over the teacher's top-k, renormalised. At k=64 that covers
99.9% of the teacher's mass on this model, so little is lost by truncating.

mlx-lm's trainer takes a custom `loss` and `iterate_batches`, and calls
``loss(model, *batch)`` — so the teacher's top-k rides along in the batch tuple
and the rest of the loop (schedule, accumulation, checkpointing, adapter
saving) is reused unchanged.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from ..config import Config
from ..logits import load_teacher_logits

BUCKET = 128


def distill_loss(model, tokens, lengths, teacher_ids, teacher_logprobs, has_teacher,
                 alpha: float = 0.3):
    """Cross-entropy against the sampled token, plus KL against the teacher.

    `tokens` is [B, T+1]; `lengths` holds (prompt_end, sequence_end) per row, as
    mlx-lm's own loss expects. `teacher_ids` and `teacher_logprobs` are
    [B, T, K], zero-padded outside the target span, and `has_teacher` is [B, T]
    marking where that padding is real data.
    """
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    logits = model(inputs)

    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    n_tokens = mask.sum()

    # Everything is derived from one [B, T] log-normaliser and two gathers, so
    # no [B, T, vocab] float32 tensor is ever materialised. The vocabulary is
    # 248k wide: forming logprobs densely costs ~1.3 GB per step at T=1280 and
    # walks the allocator into Metal's live-resource limit within ~750 steps.
    lse = mx.logsumexp(logits.astype(mx.float32), axis=-1)                  # [B, T]
    target_logit = mx.take_along_axis(logits, targets[..., None], axis=-1)
    ce = lse - target_logit.squeeze(-1).astype(mx.float32)                  # [B, T]
    ce = (ce * mask).sum() / n_tokens

    # KL(teacher || student) over the teacher's top-k, renormalised.
    student_k = mx.take_along_axis(logits, teacher_ids, axis=-1).astype(mx.float32)
    student_k = student_k - lse[..., None]                                  # [B, T, K]

    t_logprobs = teacher_logprobs.astype(mx.float32)
    t_logprobs = t_logprobs - mx.logsumexp(t_logprobs, axis=-1, keepdims=True)
    t_probs = mx.exp(t_logprobs)

    # Positions with no teacher row carry the -30 fill, which renormalises to
    # certainty on token id 0; they must be excluded rather than left to the
    # target-span mask, which does not know about them.
    kl_mask = mask * has_teacher
    kl = (t_probs * (t_logprobs - student_k)).sum(axis=-1)
    kl = (kl * kl_mask).sum() / mx.maximum(kl_mask.sum(), 1)

    return alpha * ce + (1 - alpha) * kl, n_tokens


def make_batches(dataset, teacher, tok, batch_size: int, max_seq_length: int, loop: bool, k: int):
    """Yield (tokens, lengths, teacher_ids, teacher_logprobs, has_teacher).

    Mirrors mlx-lm's `iterate_batches` but carries the teacher's top-k, aligned
    to the target span and zero-padded elsewhere.
    """
    import numpy as np

    idx = list(range(len(dataset)))
    while True:
        indices = np.random.permutation(idx) if loop else idx
        for s in range(0, len(indices) - batch_size + 1, batch_size):
            chunk = [int(i) for i in indices[s : s + batch_size]]
            seqs, spans, tk_ids, tk_lps = [], [], [], []
            for i in chunk:
                tokens, prompt_len = dataset[i]
                tokens = tokens[:max_seq_length]
                seqs.append(tokens)
                spans.append((prompt_len, len(tokens)))
                tk_ids.append(teacher[i][0] if i in teacher else None)
                tk_lps.append(teacher[i][1] if i in teacher else None)

            # Round the padded width to a bucket. Every sample otherwise has
            # its own length, so every step allocates uniquely-shaped buffers
            # and the allocator walks into Metal's live-resource limit. With
            # buckets the same few shapes are reused and recycled.
            longest = max(len(s_) for s_ in seqs)
            width = min(max_seq_length, -(-longest // BUCKET) * BUCKET)
            batch = np.zeros((len(seqs), width), np.int32)
            ids = np.zeros((len(seqs), width - 1, k), np.int32)
            lps = np.full((len(seqs), width - 1, k), -30.0, np.float32)
            cov = np.zeros((len(seqs), width - 1), np.float32)
            for b, (seq, (p_len, end)) in enumerate(zip(seqs, spans)):
                batch[b, : len(seq)] = seq[:width]
                if tk_ids[b] is None:
                    continue
                n = min(end - p_len, tk_ids[b].shape[0], width - 1 - (p_len - 1))
                if n > 0:
                    lo = p_len - 1
                    ids[b, lo : lo + n] = np.array(tk_ids[b][:n])
                    lps[b, lo : lo + n] = np.array(tk_lps[b][:n], dtype=np.float32)
                    cov[b, lo : lo + n] = 1.0
            yield (
                mx.array(batch),
                mx.array([[p, e] for p, e in spans]),
                mx.array(ids),
                mx.array(lps),
                mx.array(cov),
            )
        if not loop:
            break


class _Dataset:
    """Tokenised samples plus the index where the assistant's turn begins."""

    def __init__(self, path: Path, tok):
        self.items = []
        with path.open() as f:
            for line in f:
                messages = json.loads(line)["messages"]
                prompt = tok.apply_chat_template(
                    messages[:-1], add_generation_prompt=True, tokenize=False
                )
                full = prompt + messages[-1]["content"]
                self.items.append((tok.encode(full), len(tok.encode(prompt))))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def train(cfg: Config, alpha: float = 0.3, top_k: int = 64) -> Path:
    from functools import partial

    from mlx_lm import load
    from mlx_lm.tuner.trainer import TrainingArgs
    from mlx_lm.tuner.trainer import train as mlx_train
    from mlx_lm.tuner.utils import build_schedule, linear_to_lora_layers

    from ..gdn_chunkwise import install

    t = cfg.distill["train"]
    if t.get("gdn_chunk"):
        install(chunk=t["gdn_chunk"])
        print(f"[odistil] chunkwise gated-delta path active (C={t['gdn_chunk']})")

    model, tok = load(cfg.train_base())
    model.freeze()
    linear_to_lora_layers(
        model,
        t["num_layers"],
        {"rank": t["rank"], "scale": t["scale"], "dropout": t["dropout"]},
    )
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"Trainable parameters: {trainable / 1e6:.3f}M")

    train_set = _Dataset(cfg.path("train", "train.jsonl"), tok)
    valid_set = _Dataset(cfg.path("train", "valid.jsonl"), tok)
    teacher_train = load_teacher_logits(cfg, "train")
    teacher_valid = load_teacher_logits(cfg, "valid")
    print(f"teacher logits: {len(teacher_train)} train, {len(teacher_valid)} valid samples")
    missing = len(train_set) - len(teacher_train)
    if missing:
        print(f"  note: {missing} samples have no teacher logits; the KL term is "
              f"masked off for them and they train on CE alone")

    adapter_dir = cfg.path("adapters")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "fine_tune_type": "lora",
                "num_layers": t["num_layers"],
                "lora_parameters": {
                    "rank": t["rank"],
                    "scale": t["scale"],
                    "dropout": t["dropout"],
                },
            },
            indent=2,
        )
    )

    # mlx-lm wants the schedule as a dict; the config stores just its name,
    # same as pipeline/train.py assembles for the CLI path.
    if t.get("lr_schedule"):
        schedule = build_schedule(
            {
                "name": t["lr_schedule"],
                "warmup": t.get("warmup", 0),
                "arguments": [t["learning_rate"], t["iters"], t["learning_rate"] / 10],
            }
        )
    else:
        schedule = t["learning_rate"]
    args = TrainingArgs(
        batch_size=t["batch_size"],
        iters=t["iters"],
        val_batches=t.get("val_batches", 8),
        steps_per_report=10,
        steps_per_eval=t["steps_per_eval"],
        steps_per_save=t["save_every"],
        max_seq_length=t["max_seq_length"],
        adapter_file=str(adapter_dir / "adapters.safetensors"),
        grad_checkpoint=t["grad_checkpoint"],
        grad_accumulation_steps=t.get("grad_accumulation_steps", 1),
        # Without this the allocator creeps into Metal's 499000 live-resource
        # limit and dies mid-run; the KL term adds enough graph nodes to reach
        # it around iteration 700. The CLI path sets the same flag.
        clear_cache_threshold=t.get("clear_cache_threshold", 8),
    )

    from mlx_lm.tuner.callbacks import TrainingCallback

    class _ClearCache(TrainingCallback):
        """Metal tracks a finite number of live resources; recycle on a cadence."""

        def on_train_loss_report(self, info):
            mx.clear_cache()

    print(f"distillation loss: {alpha:.2f} * CE + {1 - alpha:.2f} * KL(top-{top_k})")
    mlx_train(
        model=model,
        optimizer=optim.AdamW(learning_rate=schedule),
        train_dataset=train_set,
        val_dataset=valid_set,
        args=args,
        loss=partial(distill_loss, alpha=alpha),
        iterate_batches=partial(
            _batches, teacher_train=teacher_train, teacher_valid=teacher_valid,
            tok=tok, k=top_k, train_set=train_set,
        ),
        training_callback=_ClearCache(),
    )
    return adapter_dir


def _batches(dataset, batch_size, max_seq_length, loop=False, seed=None, comm_group=None,
             *, teacher_train, teacher_valid, tok, k, train_set):
    """Matches mlx-lm's `iterate_batches` signature so its loop can call us."""
    teacher = teacher_train if dataset is train_set else teacher_valid
    return make_batches(dataset, teacher, tok, batch_size, max_seq_length, loop, k)


def sanity_check(cfg: Config, alpha: float = 0.3, top_k: int = 64) -> None:
    """One batch through the loss, to catch shape and alignment errors early."""
    from mlx_lm import load

    from ..mlxutil import load_tokenizer

    tok = load_tokenizer(cfg.student_base())
    ds = _Dataset(cfg.path("train", "train.jsonl"), tok)
    teacher = load_teacher_logits(cfg, "train")
    gen = make_batches(ds, teacher, tok, 1, cfg.distill["train"]["max_seq_length"], False, top_k)
    tokens, lengths, ids, lps, cov = next(gen)
    print(f"tokens {tokens.shape} | lengths {lengths.tolist()} | topk {ids.shape}")
    covered = int((lps[0, :, 0] > -30).sum())
    print(f"positions with teacher logits: {covered} of {int(lengths[0, 1] - lengths[0, 0])} target")
    model, _ = load(cfg.student_base())
    loss, n = distill_loss(model, tokens, lengths, ids, lps, cov, alpha=alpha)
    mx.eval(loss, n)
    print(f"loss {float(loss):.4f} over {int(n)} tokens")
    math.isfinite(float(loss)) or print("  WARNING: loss is not finite")
