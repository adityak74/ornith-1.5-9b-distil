# Decisions log — distilling Ornith-1.5-9B

Running record of every judgement call made while producing the distilled
model, with the evidence behind it. Newest section last.

## 0. What the two sets of numbers mean

The benchmark table this project started from was measured on the **oMLX
server**, not by this repo. Its prompt formatting, answer parsing, token budget
and truncation handling are unknown to me, so those numbers are context, not a
yardstick. Every claim this repo makes compares checkpoints measured by
`odistil eval` against each other. The oMLX numbers stay in
`benchmarks/baseline.json` for reference and are labelled as such.

Concretely: oMLX reports 91.5% HumanEval for the bf16 student; this harness
measures 82.3% for the same weights at a 2048-token budget, of which 20 items
were truncations scored as wrong and 3 were `NameError`s from missing prompt
preambles. After fixing both, the same weights are re-measured (see §3). The
two harnesses will not agree, and that is fine — what matters is that the
distilled model is measured the same way as its own baseline.

## 1. Sequence-level distillation, not logit distillation

All three models share an identical 248,044-token vocabulary (verified: the two
Ornith `tokenizer.json` files are byte-identical, Qwen's differs only in
formatting), so KL-on-logits is *possible*. Not doing it for v1:

- It needs a custom MLX training loop; `mlx_lm lora` does cross-entropy on text.
- Online KD needs teacher + student co-resident (19 + 18 GB) plus optimizer
  state, against a 52 GB recommended working set.
- The offline variant (store teacher top-k logprobs during generation) is the
  right design, but it doubles the cost of an already long generation stage.

v1 is teacher traces + rejection sampling, which is where most of the gain is.
Offline top-k logprobs are the first thing to add for v2.

## 2. LoRA over the bf16 base, not a full fine-tune

A 9B in bf16 is ~18 GB of weights; AdamW adds ~72 GB of fp32 state. That does
not fit in 64 GB. Rank-64 LoRA on all layers over the **full-precision** base,
fused afterwards, is the largest thing that fits. Training in bf16 and
quantizing last also means the quantizer starts from an improved checkpoint
rather than compounding two lossy steps.
