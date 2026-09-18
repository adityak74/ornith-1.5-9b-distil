"""Stage 4b -- preference training on termination pairs (v10).

Direct preference optimisation (arXiv:2305.18290) over the pairs that
`pairs` built: the teacher's verified trace is preferred to the student's own
failed trace on the same prompt. With the reference fixed at the model we
start from,

    L = -log sigmoid( beta * [ (log pi(c) - log ref(c)) - (log pi(r) - log ref(r)) ] )
        + sft * CE(c)

The CE term on the chosen trace keeps the policy near data it should still
produce (the RPO regulariser, arXiv:2404.19733); without it DPO is free to
lower both sides' likelihood.

Memory is the constraint here, not the maths. A pair is two sequences of up
to 2,048 tokens and both cannot be in one backward graph on 64 GB. The DPO
gradient factorises:

    dL/dtheta = -beta * sigmoid(-delta) * ( d log pi(c) - d log pi(r) )

so the step is: one no-grad forward on the rejected trace, one forward/
backward on the chosen trace with the weight sigmoid(-delta) held constant,
one forward/backward on the rejected trace with the same constant. Peak
memory equals SFT at the same length. mlx-lm's trainer cannot express a
two-graph step, so this stage runs its own loop and writes adapters in the
same format, so `fuse` and everything after it are unchanged.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten, tree_map

from ..config import Config

BUCKET = 128


def _pad(tokens: list[int], cap: int) -> mx.array:
    """Bucket the width so the allocator reuses a few shapes (see distill_train)."""
    tokens = tokens[:cap]
    width = min(cap, -(-len(tokens) // BUCKET) * BUCKET)
    return mx.array([tokens + [0] * (width - len(tokens))]), len(tokens)


def seq_logprob(model, tokens: mx.array, prompt_len: int, end: int):
    """Sum of log pi(target span) and the per-token mean, from one [T] log-normaliser.

    Never forms [T, vocab] in float32: the vocabulary is 248k wide.
    """
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    logits = model(inputs)
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= prompt_len, steps <= end - 1).astype(mx.float32)[None]
    lse = mx.logsumexp(logits.astype(mx.float32), axis=-1)
    tl = mx.take_along_axis(logits, targets[..., None], axis=-1).squeeze(-1).astype(mx.float32)
    lp = ((tl - lse) * mask).sum()
    return lp, lp / mx.maximum(mask.sum(), 1)


class _Pairs:
    def __init__(self, path: Path, tok, cap: int):
        self.items = []
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                prompt = tok.apply_chat_template(r["messages"], add_generation_prompt=True, tokenize=False)
                p_len = len(tok.encode(prompt))
                c = tok.encode(prompt + r["chosen"])
                rj = tok.encode(prompt + r["rejected"])[:cap]
                if p_len >= cap - 8:
                    continue
                self.items.append((r["id"], c, rj, p_len, bool(r.get("rejected_unterminated"))))

    def __len__(self):
        return len(self.items)


def _reference(model, pairs: _Pairs, cap: int, cache: Path) -> dict[str, tuple[float, float]]:
    """log ref(c), log ref(r) per pair with adapters absent; cached on disk."""
    ref = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [it for it in pairs.items if it[0] not in ref]
    print(f"[pref] reference log-probs: {len(todo)} to score ({len(ref)} cached)")
    for n, (pid, c, rj, p_len, _) in enumerate(todo, 1):
        tc, ec = _pad(c, cap)
        tr, er = _pad(rj, cap)
        lc, _ = seq_logprob(model, tc, p_len, ec)
        lr, _ = seq_logprob(model, tr, p_len, er)
        mx.eval(lc, lr)
        ref[pid] = (float(lc), float(lr))
        if n % 20 == 0:
            cache.write_text(json.dumps(ref))
            mx.clear_cache()
            print(f"  {n}/{len(todo)}", end="\r", flush=True)
    cache.write_text(json.dumps(ref))
    return ref


def _step(model, it, ref, cap: int, beta: float, sft: float):
    """One pair; returns (grads, stats). See the module docstring for why it is three passes.

    The model must be in train mode throughout: the gated-delta layer takes the
    fused Metal kernel (which has no vjp) in eval mode and the chunkwise form in
    train mode, and the two disagree by ~0.003 nat/token -- enough to bias delta
    if the reference were scored on one path and the policy on the other.
    """
    pid, c, rj, p_len, _ = it
    tc, ec = _pad(c, cap)
    tr, er = _pad(rj, cap)
    ref_c, ref_r = ref[pid]

    lp_r0, _ = seq_logprob(model, tr, p_len, er)          # no grad
    mx.eval(lp_r0)
    lp_r0 = float(lp_r0)

    def f_c(m):
        lp_c, mean_c = seq_logprob(m, tc, p_len, ec)
        delta = beta * ((mx.stop_gradient(lp_c) - ref_c) - (lp_r0 - ref_r))
        w = mx.sigmoid(-delta)                              # constant
        return -beta * w * lp_c - sft * mean_c, (lp_c, delta)

    (_, (lp_c, delta)), g_c = nn.value_and_grad(model, f_c)(model)
    mx.eval(g_c, lp_c, delta)
    w = float(mx.sigmoid(-delta))

    def f_r(m):
        lp_r, _ = seq_logprob(m, tr, p_len, er)
        return beta * w * lp_r

    _, g_r = nn.value_and_grad(model, f_r)(model)
    grads = tree_map(lambda a, b: a + b, g_c, g_r)
    d = float(delta)
    return grads, {"loss": math.log1p(math.exp(-d)), "margin": d, "acc": float(d > 0),
                   "lp_c": float(lp_c), "lp_r": lp_r0}


def _evaluate(model, pairs: _Pairs, ref, cap: int, beta: float) -> dict:
    tot = {"loss": 0.0, "acc": 0.0, "margin": 0.0}
    for it in pairs.items:
        pid, c, rj, p_len, _ = it
        tc, ec = _pad(c, cap)
        tr, er = _pad(rj, cap)
        lc, _ = seq_logprob(model, tc, p_len, ec)
        lr, _ = seq_logprob(model, tr, p_len, er)
        mx.eval(lc, lr)
        d = beta * ((float(lc) - ref[pid][0]) - (float(lr) - ref[pid][1]))
        tot["loss"] += math.log1p(math.exp(-d)); tot["acc"] += d > 0; tot["margin"] += d
    n = max(len(pairs), 1)
    return {k: v / n for k, v in tot.items()}


def train(cfg: Config) -> Path:
    from mlx_lm import load
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.tuner.utils import build_schedule, linear_to_lora_layers

    from ..gdn_chunkwise import install

    t = cfg.distill["train"]
    p = cfg.distill["pref"]
    beta, sft = p.get("beta", 0.1), p.get("sft_weight", 0.0)
    cap = t["max_seq_length"]
    if t.get("gdn_chunk"):
        install(chunk=t["gdn_chunk"])
        print(f"[odistil] chunkwise gated-delta path active (C={t['gdn_chunk']})")

    model, tok = load(cfg.train_base())
    model.freeze()
    model.train()   # one numerical path for reference, no-grad and grad passes (see _step)
    train_pairs = _Pairs(cfg.path("train", "pairs-train.jsonl"), tok, cap)
    valid_pairs = _Pairs(cfg.path("train", "pairs-valid.jsonl"), tok, cap)
    print(f"[pref] {len(train_pairs)} train pairs, {len(valid_pairs)} valid, beta={beta}, sft={sft}")

    # reference = the model we start from, before any adapter exists
    ref = _reference(model, train_pairs, cap, cfg.path("train", "pairs-ref.json"))
    ref.update(_reference(model, valid_pairs, cap, cfg.path("train", "pairs-ref-valid.json")))

    linear_to_lora_layers(model, t["num_layers"],
                          {"rank": t["rank"], "scale": t["scale"], "dropout": t["dropout"]})
    if t["grad_checkpoint"]:
        grad_checkpoint(model.layers[0])
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"Trainable parameters: {trainable / 1e6:.3f}M")

    adapter_dir = cfg.path("adapters")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora", "num_layers": t["num_layers"],
        "lora_parameters": {"rank": t["rank"], "scale": t["scale"], "dropout": t["dropout"]},
    }, indent=2))

    schedule = build_schedule({"name": t["lr_schedule"], "warmup": t.get("warmup", 0),
                               "arguments": [t["learning_rate"], t["iters"], t["learning_rate"] / 10]}) \
        if t.get("lr_schedule") else t["learning_rate"]
    opt = optim.AdamW(learning_rate=schedule)
    rng = random.Random(cfg.distill["seed"])
    order: list = []
    log = cfg.path("adapters", "pref-log.jsonl").open("a")
    acc: dict = {}
    t0 = time.time()
    for step in range(1, t["iters"] + 1):
        if not order:
            order = list(train_pairs.items); rng.shuffle(order)
        grads, st = _step(model, order.pop(), ref, cap, beta, sft)
        if p.get("max_grad_norm"):
            grads, _ = optim.clip_grad_norm(grads, p["max_grad_norm"])
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        for k, v in st.items():
            acc[k] = acc.get(k, 0.0) + v
        if step % t.get("clear_cache_threshold", 8) == 0:
            mx.clear_cache()
        if step % 10 == 0:
            rep = {k: v / 10 for k, v in acc.items()}
            print(f"Iter {step}: loss {rep['loss']:.3f}, margin {rep['margin']:+.2f}, acc {rep['acc']:.2f}, "
                  f"lr {float(opt.learning_rate):.2e}, {(time.time() - t0) / 10:.1f}s/it")
            log.write(json.dumps({"step": step, **rep}) + "\n"); log.flush()
            acc, t0 = {}, time.time()
        if step % t["steps_per_eval"] == 0 or step == t["iters"]:
            v = _evaluate(model, valid_pairs, ref, cap, beta)
            print(f"Iter {step}: Val loss {v['loss']:.3f}, margin {v['margin']:+.2f}, acc {v['acc']:.2f}")
            log.write(json.dumps({"step": step, "valid": v}) + "\n"); log.flush()
        if step % t["save_every"] == 0 or step == t["iters"]:
            weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(adapter_dir / "adapters.safetensors"), weights)
            mx.save_safetensors(str(adapter_dir / f"{step:07d}_adapters.safetensors"), weights)
    return adapter_dir


def sanity_check(cfg: Config) -> None:
    """One pair through reference scoring and one training step."""
    from mlx_lm import load
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from ..gdn_chunkwise import install

    t = cfg.distill["train"]; p = cfg.distill["pref"]
    cap = t["max_seq_length"]
    if t.get("gdn_chunk"):
        install(chunk=t["gdn_chunk"])   # the fused kernel has no vjp
    model, tok = load(cfg.train_base())
    model.freeze()
    model.train()
    pairs = _Pairs(cfg.path("train", "pairs-train.jsonl"), tok, cap)
    # the longest pair, so the check is also the memory check
    pairs.items = [max(pairs.items, key=lambda it: len(it[1]) + len(it[2]))]
    ref = _reference(model, pairs, cap, cfg.path("train", "pairs-ref-sanity.json"))
    print(f"reference: {ref}")
    linear_to_lora_layers(model, t["num_layers"], {"rank": t["rank"], "scale": t["scale"], "dropout": 0.0})
    if t["grad_checkpoint"]:
        grad_checkpoint(model.layers[0])
    grads, st = _step(model, pairs.items[0], ref, cap, p.get("beta", 0.1), p.get("sft_weight", 0.0))
    mx.eval(grads)
    print({k: round(v, 4) for k, v in st.items()})
    # adapters are zero-initialised, so the policy equals the reference and delta must be 0
    assert abs(st["margin"]) < 1e-2, st
    print(f"grad norm {float(optim.clip_grad_norm(grads, 1e9)[1]):.4f}")
    print(f"peak memory {mx.get_peak_memory() / 1e9:.1f} GB")
