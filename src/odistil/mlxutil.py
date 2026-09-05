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
    truncated: bool = False  # ran out of budget before closing </think>


def opens_think(rendered_prompt: str) -> bool:
    """True when the chat template already emitted an unclosed <think>."""
    return rendered_prompt.rstrip().endswith("<think>") or (
        "<think>" in rendered_prompt and "</think>" not in rendered_prompt.rsplit("<think>", 1)[1]
    )


def split_think(raw: str, pre_opened: bool = False) -> tuple[str, str | None]:
    """Separate reasoning from the answer.

    The Ornith/Qwen chat template opens the reasoning block itself -- the prompt
    ends with '<think>\n' -- so generated text usually starts *inside* the block
    and carries only the closing tag. Handle both shapes, plus the truncated
    case where the token budget ran out before the model stopped reasoning.
    """
    raw = raw.strip()
    m = THINK_RE.search(raw)
    if m:  # a full <think>...</think> pair somewhere in the output
        return THINK_RE.sub("", raw).strip(), m.group(1).strip()
    if "</think>" in raw:  # template pre-opened the block
        think, _, answer = raw.partition("</think>")
        return answer.strip(), think.removeprefix("<think>").strip()
    if "<think>" in raw:  # opened but never closed: truncated mid-reasoning
        return "", raw.split("<think>", 1)[1].strip()
    if pre_opened:
        # The prompt opened the block and nothing closed it: the model ran out
        # of budget while reasoning. There is no answer -- do not mistake the
        # reasoning text for one.
        return "", raw
    return raw, None


@lru_cache(maxsize=1)  # one model resident at a time: 64 GB unified memory
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
    pre = opens_think(text)
    answer, thought = split_think(raw, pre)
    return Completion(answer, thought, raw, len(tokenizer.encode(raw)), dt, not answer)


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
            a, th = split_think(raw, opens_think(t))
            out.append(Completion(a, th, raw, len(tokenizer.encode(raw)), time.time() - t0, not a))
        return out

    out: list[Completion] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        t0 = time.time()
        # batch_generate takes token ids, not strings.
        ids = [tokenizer.encode(t) for t in chunk]
        res = _batch(model, tokenizer, prompts=ids, max_tokens=max_tokens, sampler=sampler, verbose=False)
        dt = (time.time() - t0) / max(len(chunk), 1)
        raws = res.texts if hasattr(res, "texts") else list(res)
        for rendered, raw in zip(chunk, raws, strict=True):
            a, th = split_think(raw, opens_think(rendered))
            out.append(Completion(a, th, raw, len(tokenizer.encode(raw)), dt, not a))
    return out


def load_tokenizer(path: str):
    """Tokenizer only -- never pulls the weights into memory."""
    from pathlib import Path

    from mlx_lm.utils import hf_repo_to_path
    from mlx_lm.utils import load_tokenizer as _lt

    p = Path(path).expanduser()
    return _lt(p if p.exists() else hf_repo_to_path(path))


def tokenizer_fingerprint(path: str) -> dict:
    """Used to decide whether true logit distillation is even possible."""
    tok = load_tokenizer(path)
    probe = "def solve(x):\n    return x ** 2  # café ✅"
    return {
        "vocab_size": tok.vocab_size,
        "bos": tok.bos_token,
        "eos": tok.eos_token,
        "probe_ids": tok.encode(probe),
    }
