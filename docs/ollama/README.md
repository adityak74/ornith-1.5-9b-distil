# Ornith-1.5-9B-MLX-distil

A distilled [Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B-MLX) reasoning model for Apple Silicon, packaged for Ollama's MLX engine. On MMLU and HumanEval it matches the full-precision bf16 parent at under half the size.

```bash
ollama run adityakarnam/Ornith-1.5-9B-MLX-distil
```

Requires Ollama with the MLX engine (macOS on Apple Silicon). Thinking is on by default.

## Benchmarks

All rows measured on the same oMLX harness under the same protocol: MMLU on a fixed 1,000-question sample, TruthfulQA MC1 on all 817, HumanEval on all 164, thinking on, greedy decoding.

| Build | Size | MMLU | TruthfulQA | HumanEval |
|---|---:|---:|---:|---:|
| **This model (Ollama int4)** | **7.9 GB** | **82.6** | 77.7 | **91.5** |
| Distilled, oQ4 ([Hugging Face](https://huggingface.co/adityak74/Ornith-1.5-9B-MLX-distil-oQ4)) | 4.9 GB | 83.5 | 79.0 | 90.8 |
| Ornith-1.5-9B, oQ4 (stock) | 4.9 GB | 78.0 | 80.7 | 87.8 |
| Ornith-1.5-9B, oQ8 (stock) | 8.9 GB | 83.1 | 80.7 | 88.4 |
| Ornith-1.5-9B, bf16 (parent) | 17 GB | 82.6 | 80.5 | 91.5 |

- Against the stock 4-bit build: **+4.6 MMLU and +3.7 HumanEval**.
- Against the bf16 parent: identical on MMLU (826/1000) and HumanEval (150/164) at 46% of the size.
- TruthfulQA is 2.8 points below the parent. Every distilled build shows this, and the cause is fewer "I don't know" answers; see the [retrospective](https://github.com/adityak74/ornith-1.5-9b-distil/blob/main/RETROSPECTIVE.md).
- The difference from the oQ4 build is within about one standard error on all three benchmarks.

## oQ4 vs int4 vs bf16

The same distilled weights at two quantizations, against the full-precision parent. Correct counts, then accuracy, from the same oMLX runs.

| | Distilled oQ4 | **Distilled int4 (this model)** | bf16 parent |
|---|---:|---:|---:|
| Size on disk | 4.9 GB | **7.9 GB** | 17 GB |
| Quantization | 4-bit, 120 modules at 5/6/8-bit | int4, 49 tensors int8, 226 bf16 | none |
| Runs on | oMLX, mlx-lm | **Ollama (MLX engine)**, oMLX, mlx-lm | oMLX, mlx-lm |
| MMLU (1,000) | 835 · 83.5% | **826 · 82.6%** | 826 · 82.6% |
| TruthfulQA (817) | 645 · 79.0% | **635 · 77.7%** | 658 · 80.5% |
| HumanEval (164) | 149 · 90.8% | **150 · 91.5%** | 150 · 91.5% |
| Wall clock, all three | 7.7 h | **8.5 h** | 18.4 h |

- **int4 against oQ4:** −0.9 MMLU, −1.3 TruthfulQA, +0.7 HumanEval (one problem). One standard error is about 1.2, 1.5 and 2.2 points respectively, so neither build is measurably better. int4 is 60% larger and about 10% slower.
- **int4 against bf16:** identical correct counts on MMLU and HumanEval, 2.8 points lower on TruthfulQA, at 46% of the size and 2.2× faster end to end.
- **Which to use:** int4 if you run Ollama; oQ4 if you run oMLX or mlx-lm and want the smallest file. Both are the same trained model.

## What this build is

The weights are the distilled model (v1): two-teacher sequence-level distillation from Ornith-1.5-35B-A3B (code) and Qwen3.6-35B-A3B (knowledge and truthfulness), with rejection sampling so only verified-correct teacher traces were trained on. Training was LoRA rank 32 over the bf16 base on one 64 GB M4 Pro, fused afterwards.

Quantization is Ollama's own MLX recipe applied to those bf16 weights: 152 tensors at int4, 49 at int8 (including the output head and some attention and MLP projections), and 226 kept in bf16 (embeddings, norms, gated-delta parameters), all at group size 64.

This is not the oQ4 build published on Hugging Face. oQ4 promotes 120 modules to 5- and 6-bit, which Ollama's MLX format cannot represent (it supports int4, int8, nvfp4, mxfp4 and mxfp8), so the oQ4 files will not load here. This build keeps more precision in more places, which is why it is larger and why it ties the parent on two of three benchmarks.

## Getting the most out of it

This model's main weakness, like most quantized reasoning models, is that it sometimes keeps reasoning past the point where more reasoning helps, and runs out of budget before answering.

- **Give it room.** Don't cap `num_predict` low on hard problems. With a 2,048-token cap, the oQ4 build of these weights ran out of budget before answering on 24 of 164 HumanEval problems.
- **Soft HALT is not available in Ollama.** The project's fix is a small bias on the `</think>` token that grows the longer reasoning runs. On the oQ4 build it lifted HumanEval from 130 to 147 of 164 at a 2,048-token budget with no regressions, and it helped every model tested, full precision included. It needs a logits processor, which Ollama does not expose. To use it, run the MLX build with the project's harness: `odistil eval --think-bias 1024:0.02`.

## Parameters

| | |
|---|---|
| Architecture | `qwen3_5` hybrid: gated DeltaNet linear attention plus full attention |
| Parameters | 9.4B |
| Context | 262,144 tokens |
| Capabilities | completion, tools, thinking |
| System prompt | `You are Ornith, a careful and concise assistant.` |

## Links

- Code, per-item evaluation records, and a 54-section decision log: https://github.com/adityak74/ornith-1.5-9b-distil
- oQ4 build for oMLX / mlx-lm: https://huggingface.co/adityak74/Ornith-1.5-9B-MLX-distil-oQ4
- Paper: *HALT: Quantized Reasoning Models Don't Stop, and a Harness-Applied Limit on Thinking Repairs Them* (arXiv link to follow)
- Base model: [ornith-ai/Ornith-1.5-9B-MLX](https://huggingface.co/ornith-ai/Ornith-1.5-9B-MLX)

Built by Aditya Karnam Gururaj Rao.
