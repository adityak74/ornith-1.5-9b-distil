---
language: en
library_name: mlx
pipeline_tag: text-generation
base_model: ornith-ai/Ornith-1.5-9B-MLX
tags:
- mlx
- apple-silicon
- distillation
- knowledge-distillation
- code
- quantized
- qwen3_5
- conversational
---

# Ornith-1.5-9B-MLX-distil-oQ4

**A 4.9 GB model that outscores the 8.9 GB one it's based on — and matches a 17 GB
model on MMLU.**

Distilled from two teachers, then quantized with the same mixed-bit **oQ4** scheme
as the stock release. Runs on any Apple Silicon Mac with ~8 GB of free memory.

| | MMLU | TruthfulQA | HumanEval | size |
|---|---:|---:|---:|---:|
| **Ornith-1.5-9B-distil-oQ4** ← *this model* | **83.5%** | 79.0% | **90.8%** | **4.9 GB** |
| Ornith-1.5-9B-oQ4 *(what it replaces)* | 78.0% | 80.7% | 87.8% | 4.9 GB |
| Ornith-1.5-9B-oQ8 | 83.1% | 80.7% | 88.4% | 8.9 GB |
| Ornith-1.5-9B bf16 *(full precision)* | 82.6% | 80.5% | 91.5% | 17 GB |

**+5.5 MMLU, +3.0 HumanEval** over the oQ4 it replaces — same size, same bit width.
It also beats the **1.8× larger** oQ8 on both, and edges out the **3.5× larger**
bf16 base on MMLU while landing 0.7 points behind it on HumanEval.

It got faster too: **HumanEval −33%, MMLU −15%** wall clock versus the stock oQ4,
because it stops reasoning when it's done instead of rambling into the token limit.

> ⚠️ **TruthfulQA is 1.7 points worse** than the stock oQ4 (79.0% vs 80.7%),
> reproduced on two independent harnesses. See [Limitations](#limitations).

---

## Quick start

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("adityak74/Ornith-1.5-9B-MLX-distil-oQ4")

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Write a Python function to check if a string is a palindrome."}],
    add_generation_prompt=True,
    tokenize=False,
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=1024))
```

Or from the command line:

```bash
mlx_lm.generate --model adityak74/Ornith-1.5-9B-MLX-distil-oQ4 \
  --prompt "Explain the difference between a process and a thread." --max-tokens 1024
```

This is a **reasoning model** — it thinks before answering, and the chat template
opens a `<think>` block for it. Give it room: **4096 max tokens for coding
problems**, 2048+ for everything else. Truncating mid-reasoning throws away the
answer entirely.

## Which variant should I use?

| your situation | pick |
|---|---|
| Coding, general assistant, or knowledge work under ~8 GB | **this model** |
| Truthfulness and resisting misconceptions matter most | stock `oQ4` or `oQ8` |
| You have 20 GB+ free and want the best 9B code quality | `Ornith-1.5-9B` bf16 |
| You have 20 GB+ and want the best overall | `Ornith-1.5-35B-A3B-4bit` — faster too, being a 3B-active MoE |

## How it was made

Two teachers, split by what each is measurably good at:

- **[Qwen3.6-35B-A3B-4bit](https://huggingface.co/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit)** → knowledge and truthfulness (89.3% MMLU)
- **Ornith-1.5-35B-A3B-4bit** → code (93.3% HumanEval, and 4.6× faster to sample from)

**Rejection sampling did the heavy lifting.** 3,974 teacher traces were generated
and only *verifiably correct* ones kept — multiple-choice answers matched against
the gold letter, open questions against gold aliases, and **generated code executed
against its source dataset's real tests**. 1,672 survived. Anything sharing a
13-gram with MMLU test, TruthfulQA or HumanEval was dropped before training.

Training was **rank-32 LoRA over the top 16 layers of the bf16 base** (3,200 steps,
1.76M tokens on one M4 Pro), fused back to full precision, and only then quantized —
so the quantizer starts from an improved checkpoint instead of compounding two
lossy steps. The oQ4 map (4-bit affine, group size 64, with 120 modules promoted to
5/6/8 bits) was replicated exactly from the stock release, so the comparison above
is like-for-like.

### Why it works: termination, not knowledge

The interesting part. On a strict harness that scores unfinished output as wrong,
the stock oQ4 **fails to reach an answer on 24% of HumanEval problems** — it
reasons past the token budget. This model cuts that to 12%, and **22 of the 28
problems it gains are ones where the stock model simply ran out of road**.

Quantization appears to damage a model's ability to *stop reasoning* more than its
ability to *reason*. Training on teacher traces that terminate cleanly repairs
much of it. That also explains the flat multiple-choice results: a truncated
multiple-choice answer is often still recoverable, a truncated program never is.

## Limitations

- **TruthfulQA regressed 1.7 points.** The truthfulness training data came from
  TriviaQA and SciQ, which reward confident factual recall — close to the opposite
  of what TruthfulQA measures, which is resisting a plausible-sounding falsehood.
  This is a known defect with a known fix, not a mystery.
- **MMLU and HumanEval gains are measured, TruthfulQA aside.** Both numbers above
  come from the same harness across every row, but a single harness is a single
  harness. Independent evaluation is welcome.
- **It is a 9B model.** It does not match the 35B teachers on truthfulness
  (−7.4 points against Ornith-35B), and it is *slower* than those MoE models on
  Apple Silicon despite being smaller, because it is dense and they activate ~3B
  parameters per token. It wins on memory, not latency.
- Reasoning length means it wants a generous token budget. Small `max_tokens` will
  hurt it more than a non-reasoning model.

## Reproducing this

The full pipeline is open source, including the decisions log with every
measurement, dead end and hardware limit hit along the way:

**[github.com/adityak74/ornith-1.5-9b-distil](https://github.com/adityak74/ornith-1.5-9b-distil)**

```bash
odistil prompts    # build the prompt pool
odistil teach      # generate teacher traces
odistil dataset    # verify, decontaminate, mix
odistil train      # LoRA distillation
odistil fuse       # fuse to bf16
odistil quantize --variant oq4
odistil eval --model runs/v1/quant/oq4
```

## License and attribution

Derived from [`ornith-ai/Ornith-1.5-9B-MLX`](https://huggingface.co/ornith-ai/Ornith-1.5-9B-MLX),
which per its own model card was built on Qwen3.5 and Gemma4. **The base model card
states no license**, so the terms governing this derivative are whatever govern that
model and its upstreams — including the Gemma terms, which flow downstream. Please
check those before redistribution or commercial use. This repository claims no
rights beyond the distillation recipe itself.

Teachers: Qwen3.6-35B-A3B (Alibaba / Qwen team) and Ornith-1.5-35B-A3B (ornith.ai).
Built with [mlx-lm](https://github.com/ml-explore/mlx-lm).
