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

## 3. Harness defects found by measuring the bf16 base (and why it was worth it)

Running the bf16 student through this harness produced 82.3% HumanEval against
oMLX's 91.5%. Of the 29 failures, 22 were the harness:

- **20 truncations.** Items hit the 2048-token cap mid-reasoning and scored
  wrong. Failures averaged 1,821 generated tokens against 708 for successes.
  Fix: per-benchmark budgets (HumanEval 4096, MMLU 3072, others 2048), recorded
  in every summary, with the truncation count reported next to the accuracy.
- **3 `NameError`s.** The model returns just the function, dropping the imports
  and helpers (`encode_cyclic`, `encode_shift`) that the HumanEval prompt
  supplies. Fix: prepend the problem prompt as a preamble before the extracted
  code; the model's own definition follows the stub and wins. Rescoring the
  existing outputs with this alone moved 132/160 to 135/160.

If oMLX capped generation the same way, part of its MMLU numbers (78.0 oQ4 /
82.6 bf16) may be budget starvation rather than missing knowledge. Unknowable
from here — but it is why the distilled model is compared only against
baselines re-measured by this harness.

## 4. Data sizing: 7,284 prompts, generated in priority order

Full pool: 3,400 knowledge MCQ, 2,000 truthfulness (1,200 TriviaQA + 800 SciQ),
1,884 code (384 MBPP with tests + 1,500 CodeFeedback). Sized against measured
throughput rather than an ideal — the teachers run at roughly 6-15 s per trace,
so the full pool is a 12-15 hour generation job.

Generation order is code first, then knowledge/truthfulness. Code uses the fast
teacher (Ornith-35B), it is the target closest to reach, and it is the go/no-go
signal: if the fused model's HumanEval does not clear the bf16 base measured on
this harness, the recipe is wrong and no amount of Qwen hours will fix it.
Both stages are resumable and append-only, so training can start on whatever
has been generated when the budget runs out.

## 5. A parse check for the one slice with no gold answer

CodeFeedback prompts are free-form instructions with no tests, so the
rejection-sampling filter has nothing to check them against. Rather than accept
them unverified, they must yield an extractable, syntactically valid Python
block (`ast.parse`). That is weak, but it removes truncated traces and
prose-only answers, which are the failure modes that actually poison a student.
Everything else — MCQ letters, QA aliases, MBPP tests — is verified against
gold or by execution.

## 6. Dropped CodeFeedback; MBPP test/validation used as training data

Measured after 512 code traces, rather than assumed:

| source | traces | usable | how verified |
|---|---:|---:|---|
| MBPP | 384 | **74%** | executed against the dataset's own tests |
| CodeFeedback | 128 | **24%** | syntax only — 53% truncated to nothing |

CodeFeedback is free-form instruction data, so its answers are long, frequently
run past the 2048-token budget, and cannot be checked against anything. Three
quarters of the compute spent on it was producing nothing usable.

Dropped it, and replaced the volume with **the rest of MBPP**: our code
benchmark is HumanEval, so MBPP's test (500) and validation (90) splits are
training data, not eval data. That takes the code slice from 384 verified plus
1,500 unverifiable prompts to 974 prompts that all ship with executable tests,
for less total generation time. The decontamination pass still runs against
HumanEval regardless.

Also switched to a per-source RNG seed (`seed:dataset:split`), so re-sizing one
pool no longer reshuffles the others — without it, every trace already
generated would have stopped matching by id and been regenerated.

## 7. Generation budget: ~3,000 traces, not 7,284

Measured throughput is 9-14 s per trace (the teachers reason for a median of
~900 tokens). The original 7,284-prompt pool was a 15-20 hour generation job.
Cut to 3,974 prompts — 2,000 knowledge, 1,000 truthfulness, 974 code — which is
roughly 10 hours and, after rejection sampling, should yield ~2,700 verified
training samples. That is enough for 2-3 epochs of rank-64 LoRA, which is the
regime where this kind of distillation gets most of its gain.

## 8. Baseline established (this harness, not oMLX)

**bf16 base: 145/164 = 88.4% HumanEval**, 15 items still truncated at the 4096
budget. That is the number the distilled model has to beat; oMLX's 91.5% for
the same weights is a different measurement and is not the target.

## 9. GPU jobs must run one at a time

Running Qwen generation and LoRA training together killed both with
`[METAL] Command buffer execution failed: Insufficient Memory`. A 4-bit 35B
teacher is ~19 GB and a 9B training job peaks near 48 GB, against a 52 GB
recommended working set. Every stage from here runs serially.

## 10. Training envelope: the architecture, not the parameter count, sets it

Measured peak memory for LoRA on the 9B, batch 1, gradient checkpointing on:

| sequence length | layers adapted | peak memory | speed |
|---:|---:|---:|---:|
| 1024 | 32 (all) | 48.6 GB | 36 s/iter |
| 1024 | 16 | 47.8 GB | 20 s/iter |
| 1024 | 8 | 47.4 GB | 14 s/iter |
| 512 | 8 | 27.2 GB | 7 s/iter |

Two things follow. Memory is driven by the **forward** pass, not by how many
layers carry adapters — roughly 20 MB per token, because the hybrid GDN
`linear_attn` layers materialise per-timestep state. And the number of adapted
layers changes speed, not footprint, because it only truncates backprop.

Settled on **rank 32, top 16 of 32 layers, 896-token sequences, batch 1 with
4-step gradient accumulation**. 896 keeps peak near 42 GB, leaving headroom
against the 52 GB ceiling — at 1024 the run sits at 47.8 GB and thrashes, which
is what dropped throughput to ~30 tokens/s. Adapting the top half of the stack
is the compromise between capacity and the 2.5x slowdown of adapting all of it.

The cost of the 896-token cap: samples whose prompt plus reasoning trace exceed
it are dropped rather than truncated (a truncated target teaches the model to
stop mid-answer). That is roughly half the traces, and it biases training
toward the teachers' more concise reasoning — an acceptable trade, since the
alternative is not training on this machine at all.

## 11. Bug: the mixer sized itself from the scarcest domain

16 stray knowledge traces from the OOM-killed Qwen run caused the dataset stage
to discard 750 verified code samples, because the mixture budget was computed
from the least-supplied domain. The scale now comes from the domain that is
most over-supplied relative to its share, so every domain contributes
everything it has up to its proportional cap. Also enforced `max_seq_len`,
which was configured but never applied.

## 12. Reasoning traces kept, at the cost of half the data

At the 1024-token training cap, retention is:

| cap | with reasoning | answer only |
|---:|---:|---:|
| 896 | 43% | 98% |
| 1024 | 50% | 99% |
| 1280 | 62% | 100% |

Dropping the reasoning would keep essentially every sample. Not doing it: the
chat template pre-opens `<think>`, so a student trained on answer-only targets
learns to emit `</think>` immediately and stop reasoning — precisely the
behaviour the model is evaluated and served with. Half the data with intact
reasoning beats all of it with the reasoning removed.

Instead the knowledge teacher is now asked to **keep reasoning under 150
words**, which raises retention per GPU-hour rather than trading away the
behaviour. Rejection sampling still filters any answer that gets worse for the
brevity, so the risk is a lower keep rate, not a worse student.

## 13. Back half of the pipeline validated end to end

Before spending the ~9-hour Qwen generation, the stages after training were
proven on the code-only adapters:

- **fuse** — 17 GB bf16 checkpoint, 28 s.
- **quantize --variant oq4** — the mixed-bit map read from the shipped oQ4
  checkpoint applied cleanly: **120/120 promoted modules matched**, 4.721 bits
  per weight, 5.0 GB against the shipped model's 4.9 GB.
- **package** — loads, generates correct Python, and carries the tokenizer,
  chat template and shard index oMLX needs.

Two bugs fixed on the way: the quant predicate had the wrong arity for
`mlx_lm.convert`, and `mlx_lm fuse` needed the adapter path passed explicitly.

## 14. First result: HumanEval did not improve

| model | HumanEval (this harness, 4096 budget) | truncated |
|---|---:|---:|
| bf16 base (undistilled) | 145/164 = **88.4%** | 15 |
| distilled oQ4 | 138/164 = **84.1%** | 19 |

Item-level: the distilled model gained 7 problems and lost 14. **Eight of the
14 losses were truncations** — it ran out of budget mid-reasoning on problems
the base model finished.

That is the most useful signal in the run. Training targets were teacher traces
whose median length was ~950 tokens, so the student learned to reason at
teacher length. Longer reasoning costs accuracy at a fixed token budget, and it
also interacts badly with the 1024-token training cap: the traces that survived
the cap are not a random sample, they are the *shorter* ones, while the model
still generalised toward long reasoning.

Note this row is not yet a like-for-like comparison — the distilled model is
quantized to 4.72 bits and the baseline is bf16. The shipped-oQ4 measurement
that isolates the distillation effect is running.

Code was also the smallest slice (421 of 1,672 samples) and the one where the
teacher had least to teach: Ornith-35B is only a few points above the student
on HumanEval. The knowledge and truthfulness slices are 1,251 samples and face
much larger teacher gaps, so they are where a gain, if any, should appear.
