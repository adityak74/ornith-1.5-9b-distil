# Ornith-1.5-9B-distil

A distilled Ornith-1.5-9B: **+5.5 MMLU and +3.0 HumanEval** over the stock 4-bit release, at the same size.

| | MMLU | TruthfulQA | HumanEval |
|---|---:|---:|---:|
| **this model** | **83.5%** | 79.0% | **90.8%** |
| Ornith-1.5-9B-oQ4 (stock 4-bit) | 78.0% | 80.7% | 87.8% |
| Ornith-1.5-9B-oQ8 (8-bit, 1.8x larger) | 83.1% | 80.7% | 88.4% |
| Ornith-1.5-9B bf16 (full precision, 3.5x larger) | 82.6% | 80.5% | **91.5%** |

It beats the 1.8x larger 8-bit release on MMLU and HumanEval, and edges out the full-precision base on MMLU. It is also about 33% faster on coding tasks than the stock 4-bit model, because it stops reasoning when it is done instead of running into the token limit.

> **Note:** TruthfulQA is 1.7 points *worse* than the stock release (79.0% vs 80.7%). If truthfulness is your priority, use the stock model.

> **Quantization:** benchmark numbers above were measured on the MLX build using the oQ4 mixed-bit scheme. This Ollama build is GGUF Q4_K_M — the same distilled weights, a different quantizer — so its scores will be close but not identical.

## Usage

    ollama run adityakarnam/Ornith-1.5-9B-MLX-distil-oQ4

This is a reasoning model: it thinks before answering. Give it room — 4096 tokens for coding problems, 2048+ otherwise. Cutting it off mid-reasoning loses the answer.

## How it was made

Two teachers, split by what each is measurably better at: **Qwen3.6-35B-A3B** for knowledge and truthfulness, **Ornith-1.5-35B-A3B** for code.

3,974 teacher traces were generated and only *verifiably correct* ones kept — multiple-choice answers checked against the gold letter, open questions against gold aliases, and generated code executed against real unit tests. 1,672 survived. Anything overlapping the benchmarks by a 13-gram was dropped.

Training was rank-32 LoRA over the bf16 base, fused back to full precision, then quantized — so quantization starts from an improved checkpoint rather than compounding two lossy steps.

**Why it works:** on a strict harness the stock 4-bit model fails to reach an answer on 24% of HumanEval problems — it reasons past the token budget. This model cuts that to 12%, and 22 of the 28 problems it gains are ones where the stock model simply ran out of road. Quantization appears to damage a model's ability to *stop* reasoning more than its ability to reason.

Full pipeline and decisions log: https://github.com/adityak74/ornith-1.5-9b-distil

MLX version: https://huggingface.co/adityak74/Ornith-1.5-9B-MLX-distil-oQ4

Derived from `ornith-ai/Ornith-1.5-9B-MLX` (built on Qwen3.5 and Gemma4). The base model card states no license — check its terms before commercial use.
