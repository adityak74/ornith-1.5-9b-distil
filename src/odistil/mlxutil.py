"""Thin wrapper over mlx-lm: load once, chat-format, batch-generate, split thinking."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass
class Completion:
    text: str            # answer with the reasoning block stripped
    think: str | None    # reasoning trace, if the model emitted one
    raw: str
    tokens: int
    seconds: float


def split_think(raw: str) -> tuple[str, str | None]:
    m = THINK_RE.search(raw)
    if not m:
        # An unterminated <think> means we hit the token cap mid-reasoning.
        if "<think>" in raw:
            return "", raw.split("<think>", 1)[1]
        return raw.strip(), None
    return THINK_RE.sub("", raw).strip(), m.group(1).strip()


@lru_cache(maxsize=4)
def load(path: str):
    from mlx_lm import load as _load

    return _load(path)


def render(tokenizer, prompt: str, system: str | None = None, think: bool = True) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    kw = {"add_generation_prompt": True, "tokenize": False}
    try:
        return tokenizer.apply_chat_template(msgs, enable_thinking=think, **kw)
    except TypeError:  # template without a thinking switch
        return tokenizer.apply_chat_template(msgs, **kw)


def generate_one(
    model_path: str,
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int = 2048,
    temp: float = 0.0,
    top_p: float = 0.95,
    think: bool = True,
) -> Completion:
    import time

    from mlx_lm import generate as _generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_path)
    text = render(tokenizer, prompt, system, think)
    sampler = make_sampler(temp=temp, top_p=top_p)
    t0 = time.time()
    raw = _generate(model, tokenizer, prompt=text, max_tokens=max_tokens, sampler=sampler, verbose=False)
    dt = time.time() - t0
    answer, thought = split_think(raw)
    return Completion(answer, thought, raw, len(tokenizer.encode(raw)), dt)


def generate_batch(
    model_path: str,
    prompts: Iterable[str],
    *,
    system: str | None = None,
    max_tokens: int = 2048,
    temp: float = 0.0,
    top_p: float = 0.95,
    think: bool = True,
    batch_size: int = 8,
) -> list[Completion]:
    """Batched generation; falls back to a serial loop on older mlx-lm."""
    import time

    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_path)
    texts = [render(tokenizer, p, system, think) for p in prompts]
    sampler = make_sampler(temp=temp, top_p=top_p)

    try:
        from mlx_lm import batch_generate as _batch
    except ImportError:
        out = []
        for t in texts:
            from mlx_lm import generate as _generate

            t0 = time.time()
            raw = _generate(model, tokenizer, prompt=t, max_tokens=max_tokens, sampler=sampler, verbose=False)
            a, th = split_think(raw)
            out.append(Completion(a, th, raw, len(tokenizer.encode(raw)), time.time() - t0))
        return out

    out: list[Completion] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        t0 = time.time()
        res = _batch(model, tokenizer, prompts=chunk, max_tokens=max_tokens, sampler=sampler, verbose=False)
        dt = (time.time() - t0) / max(len(chunk), 1)
        raws = res.texts if hasattr(res, "texts") else list(res)
        for raw in raws:
            a, th = split_think(raw)
            out.append(Completion(a, th, raw, len(tokenizer.encode(raw)), dt))
    return out


def tokenizer_fingerprint(path: str) -> dict:
    """Used to decide whether true logit distillation is even possible."""
    _, tok = load(path)
    probe = "def solve(x):\n    return x ** 2  # café ✅"
    return {
        "vocab_size": len(tok),
        "bos": tok.bos_token,
        "eos": tok.eos_token,
        "probe_ids": tok.encode(probe),
    }
