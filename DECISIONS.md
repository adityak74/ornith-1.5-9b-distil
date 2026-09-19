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

## 15. Correction to §14, and the final result

§14 compared the distilled **4.72-bit** model against the **bf16** base, which
folds in the quantization cost and pointed the wrong way. The like-for-like
comparison — both models quantized with the same oQ4 map, measured by this
harness — is the one that matters:

| benchmark | shipped oQ4 | distilled oQ4 | delta | ±2 s.e. | truncations |
|---|---:|---:|---:|---:|---:|
| MMLU (250) | 81.2% | 82.4% | +1.2 pp | 6.9 | 24 → 10 |
| TruthfulQA (250) | 72.4% | 70.8% | −1.6 pp | 8.1 | 27 → 20 |
| HumanEval (164) | 72.0% | **84.1%** | **+12.2 pp** | 9.0 | 39 → 19 |

bf16 base (undistilled, unquantized) scores 88.4% HumanEval, so the distilled
oQ4 recovers most of what quantization costs.

**HumanEval is a real gain**; MMLU and TruthfulQA move within noise at n=250.
Against the shipped oQ4 the distilled model gained 28 HumanEval problems and
lost 8, and **22 of the 28 gains were problems where the shipped model ran out
of budget mid-reasoning**.

The mechanism is visible in every column: **truncations roughly halve**
(24→10, 27→20, 39→19) and mean HumanEval generation drops from 1,721 to 1,414
tokens. The shipped oQ4 rambles — 24% of its HumanEval attempts never reach an
answer — and training on teacher traces that terminate cleanly largely fixed
that. Quantization appears to damage reasoning *termination* more than
reasoning *quality*, and distillation repairs exactly that.

This also explains the flat MMLU/TruthfulQA: those are multiple-choice, where a
truncated answer is recoverable from context far more often than a truncated
program is, so there is less broken behaviour available to repair.

### What this run does not show

- No gain on knowledge or truthfulness, despite 75% of the training data
  going there. Either 1,251 samples is too few to move MMLU, or the 1024-token
  cap kept exactly the short traces that carry least knowledge.
- MMLU/TruthfulQA were measured on seeded 250-item subsets, not the full
  protocol. Deltas under ~7 pp are not resolvable at that size.

## 16. Cross-harness confirmation on oMLX

The distilled model was re-measured on the oMLX server — the same harness that
produced the original table, and the one that matters for deployment:

| model | size | MMLU | TruthfulQA | HumanEval |
|---|---:|---:|---:|---:|
| oQ4 (shipped) | 4.9 GB | 78.0% | 80.7% | 87.8% |
| oQ8 | 8.9 GB | 83.1% | 80.7% | 88.4% |
| bf16 base | 17 GB | 82.6% | 80.5% | 91.5% |
| **distil-oQ4** | **4.9 GB** | **83.5%** | 79.0% | **90.8%** |

- vs the shipped oQ4: **+5.5 MMLU, +3.0 HumanEval**, −1.7 TruthfulQA.
- vs **oQ8, which is 1.8x larger**: +0.4 MMLU, +2.4 HumanEval.
- vs the **bf16 base, 3.5x larger**: +0.9 MMLU, −0.7 HumanEval.

A 4.9 GB model that beats the 8.9 GB oQ8 on two of three benchmarks and matches
the 17 GB bf16 base within a point is the compression result this project was
after, even though none of the 86-89 targets were reached.

### The two harnesses agree on direction, and the disagreement is informative

| benchmark | this harness | oMLX |
|---|---:|---:|
| MMLU | +1.2 pp | +5.5 pp |
| TruthfulQA | −1.6 pp | −1.7 pp |
| HumanEval | +12.2 pp | +3.0 pp |

TruthfulQA agrees to within 0.1 pp. MMLU differs because this harness measured
only 250 items (±6.9 pp). HumanEval differs because the harnesses disagree
about the *shipped* model, not the distilled one: oMLX scores shipped oQ4 at
87.8% where this harness scores 72.0%. That gap is the truncation handling —
oMLX evidently recovers an answer from output this harness scores as
unfinished. So **+3.0 pp is the number to trust for deployment**, and the
+12.2 pp here was inflated by measuring a failure mode oMLX masks.

### Generation time independently confirms the mechanism

Against the shipped oQ4 on oMLX: **HumanEval −33%, MMLU −15%** (TruthfulQA
+26%). The distilled model reaches its answer sooner on the benchmarks where it
improved. That is the same termination effect seen here as halved truncation
counts, showing up on a harness that does not penalise truncation — so it is a
property of the model, not an artefact of scoring.

## 17. Why TruthfulQA regressed, and what to do about it

TruthfulQA is the one benchmark that got worse, consistently on both harnesses
(−1.6 / −1.7 pp). The likely cause is the training data: the truthfulness slice
was 626 samples from **TriviaQA and SciQ**, which reward confident factual
recall. TruthfulQA rewards the opposite reflex — resisting a plausible-sounding
falsehood and declining to assert. Training on confident recall plausibly
sharpens exactly the behaviour TruthfulQA penalises.

For v2: drop TriviaQA from the truthfulness slice and source it from data that
teaches calibration and refusal instead, or generate it — ask the teacher to
answer questions with common misconceptions and keep only traces where it
identifies the misconception. TruthfulQA itself remains eval-only.

---

# v2

## 18. What v2 changes, and why

v1 shipped at 83.5 / 79.0 / 90.8 (MMLU / TruthfulQA / HumanEval on oMLX). Two
defects were diagnosable from the v1 data, so v2 fixes exactly those rather
than changing everything at once.

**Fix 1 — the truthfulness slice taught the wrong reflex.** v1 drew it from
TriviaQA and SciQ: obscure factual recall ("in which British city are Butetown,
Splott and Roath?"). TruthfulQA measures the opposite skill — resisting a
plausible-sounding falsehood. Replaced with
[`notrichardren/misconceptions_tf`](https://huggingface.co/datasets/notrichardren/misconceptions_tf):
1,703 true/false statements, 63% of them misconceptions, with gold labels so
the traces stay verifiable. Checked for contamination: **zero 8-gram overlap
with TruthfulQA**. SciQ moves to the knowledge slice, where it always belonged.

**Fix 2 — the code teacher was never asked to be brief.** The knowledge teacher
was (§12), the code teacher was not, and v1 lost **356 code traces to the length
cap — more than it lost to wrong answers (93)**. The code slice is therefore
being regenerated with the same brevity instruction. Cost is ~2 GPU-hours,
expected return is several hundred extra verified samples in the slice that had
the fewest.

Everything else is held constant: same teachers, same rank-32/top-16 LoRA, same
1024-token cap, same oQ4 map. If v2 moves, the cause is identifiable.

## 19. A change worth making, reverted for a boring reason

Prompt sampling takes `rng.sample(range(N), n)`, which is stable when *other*
pools are resized but not when *this* pool is. Growing MMLU from 1,500 to 2,400
therefore reshuffles the selection, and every trace already generated stops
matching by id. Taking a prefix of one fixed shuffle fixes that properly.

I made the change, then measured what it cost: reusable v1 traces dropped from
2,400 to **65**. Switching methods invalidates traces generated under the old
one, and 2,400 traces is about five GPU-hours. Reverted, with the reasoning left
in the code — it is the right design to adopt at the start of a run that has
nothing to reuse, and the wrong one to adopt mid-project.

Net effect: v2 reuses all 2,400 v1 knowledge traces and generates only the
1,703 new misconception prompts and 974 regenerated code prompts.

## 20. Ollama / GGUF port: attempted, does not work

Tried publishing v1 to Ollama. It converts and loads, but generates gibberish
(`'LEDQ/[1'`). Four genuine incompatibilities were found and fixed on the way:

1. `Qwen3_5ForCausalLM` is not a recognised architecture; only the multimodal
   `Qwen3_5ForConditionalGeneration` is.
2. Tensor prefixes: the checkpoint uses `language_model.model.*`, the converter
   expects transformers-v5 `model.language_model.*` with a top-level `lm_head`.
   Fixing this resolved `token_embd.weight not found`.
3. **MLX stores conv1d as `(out, kernel, in)`; PyTorch and llama.cpp expect
   `(out, in, kernel)`.** 24 tensors needed transposing.
4. The config declares `mtp_num_hidden_layers: 1`, but the MLX checkpoint
   carries no multi-token-prediction weights, so llama.cpp expected 33 blocks
   and found 32.

After all four it loads and is numerically wrong, which points at llama.cpp's
GDN/linear-attention implementation expecting a layout MLX does not produce.
Finding it means reverse-engineering the expected layout of every tensor in the
hybrid attention path — a project, not a fix, so it was stopped here.

This is an MLX-to-GGUF problem, not a distillation problem: the fused checkpoint
is structurally identical to the stock base, which would convert equally badly.
The viable route is to convert from **PyTorch** weights instead — apply the LoRA
to a non-MLX checkpoint and use llama.cpp's own `convert_hf_to_gguf.py`, whose
per-tensor mapping is inspectable — and only if llama.cpp supports this hybrid
architecture at all.

**Process note:** running these conversions while teacher generation was live
killed the code slice at 336/974 through memory pressure — breaking the
serialisation rule from §9 that was written after the same mistake. It also
left 130 GB of orphaned blobs in `~/.ollama/models/blobs`, which `ollama rm`
does not reclaim.

## 21. v2 data: both fixes worked

**The code teacher's brevity instruction cut trace length sharply:**

| | median tokens | p75 | at 2048 cap | truncated to nothing |
|---|---:|---:|---:|---:|
| v1 (no brevity) | 1,014 | 1,912 | 229 | 118 |
| v2 (brevity) | **717** | **1,495** | **160** | **72** |

Median down 29%. Interestingly the same instruction did *not* shorten the
knowledge teacher's traces (median 950 in v1 vs 925 in v2) — code answers have
a natural stopping point once the function is written, while multiple-choice
reasoning seems to expand to fill whatever budget it is given.

**Resulting training set, against v1:**

| slice | v1 | v2 | change |
|---|---:|---:|---|
| code | 421 | **530** | +26% (from 14 *fewer* traces) |
| knowledge | 625 | **880** | +41% |
| truthfulness | 626 | **876** | +40%, and now calibration rather than recall |
| total | 1,672 | **2,286** | **+37%** |

Keep rate 45% (v2) vs 42% (v1). The misconception teacher was right on 91% of
items — the `reject:tf` count is 161 of 1,703.

The 1024-token cap is still the dominant filter at 2,338 rejections, 46% of
everything generated and far more than every wrong answer combined (361). It
remains the single highest-value thing to fix, and it is a memory limit, not a
data problem.

Training runs at the same 3,200 iterations as v1 — with 37% more data that is
1.4 epochs rather than 1.9, which also reduces the mild overfitting v1 showed
(train loss 0.138 against val 0.310).

## 22. v2 result: no improvement. Not shipped.

| benchmark | v1 | v2 | delta | ±2 s.e. | truncations |
|---|---:|---:|---:|---:|---:|
| MMLU (250) | 82.4% | 81.2% | −1.2 | 6.9 | 10 → 15 |
| TruthfulQA (250) | 70.8% | **71.6%** | +0.8 | 8.1 | 20 → 22 |
| HumanEval (164) | **84.1%** | 81.1% | −3.0 | 8.4 | 19 → 13 |

Every delta is inside its error bar and two of three lean negative, so v2 was
not uploaded. v1 remains the released model.

**The §17 hypothesis is not supported.** TruthfulQA moved +0.8 pp — nothing.
Replacing recall data with gold-labelled misconception data was a clean,
well-motivated intervention aimed at a specific diagnosis, and it did not
reproduce as a benchmark gain. The diagnosis was plausible and wrong, or the
effect is smaller than this harness can resolve at n=250.

### The confound I introduced

Holding **iterations** constant rather than **epochs** meant v2's 37% larger
dataset got 1.43 passes where v1 got 1.95. v2 is therefore both better-fed and
less-trained, and those two effects cannot be separated from this run. Matching
v1's epoch count needs 4,375 iterations (+37% compute, roughly 3 more hours).

That is the single cleanest follow-up: rerun v2's data at 4,375 iterations. If
it still does not beat v1, the data changes genuinely did not help and the
remaining lever is the token cap, not the data mixture.

### Lower validation loss did not become better benchmarks

v2 fits the teacher distribution measurably better — val loss 0.267 against
v1's 0.310, train 0.116 against 0.138 — and scores the same or slightly worse
on all three benchmarks. Fitting teacher traces more closely is not the same as
answering benchmark questions more correctly, and val loss should not be used
as a stopping signal for this kind of work.

### What did reproduce

The **termination** effect from §15 shows up again: HumanEval truncations fell
19 → 13, the benchmark where v2's data was most improved (shorter code traces).
It just did not convert into accuracy this time. MMLU and TruthfulQA
truncations rose slightly, which fits the observation in §21 that the brevity
instruction shortened code traces but not multiple-choice reasoning.

### Ranking for v3

1. **Raise the 1024-token cap.** It rejected 46% of everything generated in v2
   (2,338 traces), six times what wrong answers cost. It is a memory limit, and
   every data-mixture change so far has been small next to it.
2. **Match epochs before concluding anything about data.** See above.
3. Offline top-k logit distillation, which the shared vocabulary allows.

Data-source tuning is now the *least* promising lever: two rounds of it moved
nothing outside noise.

---

# v3 (prepared, not yet run)

## 23. QLoRA: train against the quantized forward pass

v1 and v2 both did post-training quantization — LoRA on bf16, fuse, then
quantize. Nothing in training was quantization-aware, which is an odd way to
approach a problem that §15 identified as *quantization damage*: the adapter
learns on a clean model and we hope the improvement survives being quantized.

v3 attaches the adapter to the **oQ4 checkpoint** instead (`train.base:
student:oq4`), so it sees the degraded forward pass and compensates for it
directly. This is QLoRA, not true QAT — MLX has no fake-quant/straight-through
primitives, only real quantization — but it is the variant that matters here
and `mlx_lm lora` supports it natively.

**The fuse step stays lossless in shape.** `LoRALinear.fuse(dequantize=False)`
dequantizes the layer, adds the low-rank update, and re-quantizes **at that
layer's own bits and group size**. The oQ4 mixed-bit map therefore survives
fusing untouched, and v3 needs no separate quantize stage. Requantisation does
reintroduce error on the corrected weights (the problem QA-LoRA and LoftQ
address); if that proves costly, the fallback is shipping base + adapter
separately, at the cost of a 173 MB sidecar oMLX may not load.

## 24. The cap rises because the base got smaller

A 4-bit base is 4.9 GB against bf16's 18 GB. v1/v2 peaked at 47.3 GB for
1024 tokens, of which roughly 29 GB was activations, so freeing ~13 GB of
weights should buy sequence length rather than just headroom.

Set to **1536**, which matters because the cap has been the dominant data
filter all along. Measured on v2's existing traces:

| cap | retained | samples | vs 1024 |
|---:|---:|---:|---:|
| 1024 | 48% | 2,376 | — |
| 1280 | 67% | 3,338 | +962 |
| **1536** | **79%** | **3,942** | **+1,566** |
| 2048 | 94% | 4,665 | +2,289 |

**+66% training data from traces already on disk** — no regeneration. If 1536
does not fit in memory, fall back to 1280 (+40%) before giving up on it.

## 25. Iterations set by epochs this time

§22's confound was holding iterations constant while the dataset grew, so v2
got 1.43 epochs against v1's 1.95. v3 sets **7,200 iterations**, which is ~1.95
epochs over the ~3.7k samples the 1536 cap should yield — matching v1 on the
axis that actually matters. Expect 12-20 hours depending on how QLoRA's
throughput compares; a 4-bit base moves less memory per step, so it may be
faster than bf16 per iteration.

## 26. What v3 tests, and what would falsify it

One hypothesis: *the adapter should learn against the quantization it will be
deployed under.* If v3 beats v1 on the oMLX harness, that is the answer, and it
also happens to double the training data via the raised cap — so a win is
confounded between the two changes and would need a follow-up to attribute.

If it loses, then three rounds have failed to beat v1 and the honest conclusion
is that a rank-32 LoRA over a few thousand samples has extracted what it can
from this student, and the remaining gap to the 35B teachers is capacity, not
recipe.

## 27. My TruthfulQA eval was measuring position bias, not truthfulness

TruthfulQA's `mc1_targets` lists **the correct answer first in all 817 items**.
Presenting the choices in dataset order — which is what this harness did — makes
`A` the gold answer every time. A model that always replies "A" scores 100%.

Checked against the recorded outputs: of v1's 177 correct answers, **177 were
'A' picks**. Same for v2: 179 of 179. The score was a pure measure of how often
the model answers 'A'.

Every TruthfulQA number this harness produced is therefore void — v1 70.8%,
v2 71.6%, shipped oQ4 72.4% — along with every comparison drawn from them.
Fixed by shuffling each item's choices under a per-item seed (deterministic, so
models stay comparable); gold now spans positions, 23% on 'A'. A regression
test pins it.

### This may reach further than my harness

The oMLX numbers are independent measurements, so they are not void by this bug
— **but the same trap is easy to fall into**, and it is worth checking whether
that harness shuffles. If it does not, its TruthfulQA column is measuring the
same thing, for every model in the table.

The stakes are concrete: **the TruthfulQA regression is the entire reason v2
exists.** §17 diagnosed it, §18 rebuilt the truthfulness slice around it, and
§22 recorded the failure to fix it. If the regression is an artifact of choice
ordering, then v2 was aimed at a phantom, and the "distillation costs
truthfulness" conclusion in §15 and in the published model card needs
revisiting rather than defending.

Worth noting what *would* explain the pattern under this bug: distillation
makes the student more decisive and less prone to defaulting to the first
option, which lowers an always-'A' score while leaving truthfulness unchanged
or better. That fits every observation — v1 below stock, v2 further below,
and the effect growing with training — at least as well as the forgetting story
in §26, and it is the more parsimonious explanation.

**Next measurement, once the GPU frees up:** re-run TruthfulQA with shuffled
choices for stock oQ4, v1 and v2. Until then, no truthfulness claim from this
project should be treated as established.

## 28. The TruthfulQA regression is an abstention deficit, and it is caused by rejection sampling

With per-question oMLX output for all three models, the cause is unambiguous.
Splitting TruthfulQA by whether the correct answer expresses uncertainty
("I have no comment", "it is unknown", "there is no scientific consensus"):

| model | overall | hedge-gold (117 q) | all others (700 q) | gap |
|---|---:|---:|---:|---:|
| stock oQ4 | 80.7% | **76.1%** | 81.4% | 5.4 pp |
| v1 distil | 78.9% | **65.0%** | 81.3% | **16.3 pp** |
| v2 distil | 76.9% | 67.5% | 78.4% | 10.9 pp |

**v1 matches stock on ordinary questions (81.3 vs 81.4) and loses 11 points on
questions where the right answer is "I don't know."** Restricted to those
items: 19 lost against 6 gained, p ≈ 0.016. The entire regression lives there.

The cause is the pipeline's central quality mechanism. Rejection sampling keeps
only *verifiably correct* teacher traces, so all 1,672 v1 targets are confident,
specific, correct answers — **zero examples of appropriate uncertainty**. The
student learns that the correct move is always to commit, and TruthfulQA is
largely a test of when not to. Nothing else it learned was damaged.

This also explains v2 cleanly. Its misconception slice targeted truthfulness
but consisted of 876 more confident True/False answers, so it added more of the
cause. v2 recovered slightly on hedging (67.5%) while losing on ordinary
questions (78.4%), for no net gain — and the v1→v2 difference is not
significant anyway (McNemar p = 0.11).

### Corrections to earlier sections

- §17's diagnosis (TriviaQA recall) was wrong, and §18 built v2 on it.
- §26's suggestion — that SFT erodes calibration generally with training volume
  — was also wrong. The damage is confined to abstention; general accuracy is
  untouched.
- A mid-analysis claim that the v1→v2 drop was itself a hedging effect was
  wrong too: v2 is *better* on hedge questions than v1.

Three wrong hypotheses about this metric before the per-question control
arrived. The lesson is cheap and general: **get the per-item control before
theorising, not after.**

### What v3 should do, and the prediction it makes

Add an **abstention slice**: prompts whose correct response is a refusal, so
the training set contains examples of not committing. SQuAD v2's unanswerable
questions are the verifiable source (gold = the answer is not present),
checked by confirming the response declines rather than invents.

Quantified prediction, which makes this falsifiable: closing the hedge gap to
stock's level is worth **+1.6 pp overall**, taking v1's TruthfulQA from 78.9%
to **80.5%** against stock's 80.7% — neutralising the regression while keeping
+5.5 MMLU and +3.0 HumanEval, since abstention data touches neither.

If v3 adds abstention data and TruthfulQA does not move toward 80%, this
hypothesis is wrong too and the project should stop theorising about this
metric.

## 29. Telling the code teacher to be brief cost 5.4 points of HumanEval

v2 complete, on oMLX at full protocol:

| benchmark | v1 | v2 | delta |
|---|---:|---:|---:|
| MMLU | 83.5% | 83.1% | −0.4 |
| TruthfulQA | 79.0% | 76.9% | −2.1 |
| HumanEval | **90.8%** | **85.4%** | **−5.4** |

v2 is worse on all three, and its HumanEval is **below the stock oQ4's 87.8%** —
the distillation made code worse than not distilling at all.

The likely cause is §21's "success". The code teacher was given a brevity
instruction to fit more traces under the length cap, and it worked: median
reasoning fell from 1,328 to 1,034 characters and the code slice grew from 421
to 530 samples. But **shorter teacher reasoning taught the student to think
less about code**, and code is exactly where reasoning pays — the same effect
§15 measured from the other direction, where the gain came from reasoning that
terminates *after* reaching an answer rather than before.

More data of a worse kind lost to less data of a better kind, by 5.4 points.

Acted on for v3: the brevity instruction is removed from the code teacher, and
v3 uses **v1's code traces** — same 974 prompts, long reasoning, already on
disk. The raised 1536-token cap is the right way to keep more of them; asking
the teacher to think less was not.

Caveat: v2 changed several things at once, so this is the most plausible
explanation rather than an isolated one. It is cheap to act on because v1's
traces already exist, and the prediction is testable — if v3's HumanEval
returns to ~90%, the brevity instruction was the cause.

## 30. QLoRA measured and dropped: slower and hungrier than bf16 here

The plan in §23 was that a 4.9 GB base instead of 18 GB would free memory for
longer sequences. Measured on this architecture:

| config | peak memory | speed |
|---|---:|---:|
| bf16 LoRA @ 1024 (v1, v2) | 47.3 GB | ~10 s/iter |
| QLoRA @ 1280 | **49.0 GB** | **~26 s/iter** |
| QLoRA @ 1536 | OOM | — |

QLoRA is **2.6x slower and uses *more* memory**. Dequantizing weights on every
forward and backward pass costs compute and temporaries that outweigh the
smaller weights entirely — and the activation memory that actually dominates is
unchanged, since activations stay bf16 either way. The planned 7,800 iterations
would have taken 56 hours.

Dropped. The upside was theoretical; the cost is measured. Dropping it also
makes v3 a clean test of the abstention hypothesis instead of two confounded
changes, which after two null results is worth more than the experiment I gave
up. QLoRA is now a *worse* idea than it looked, not a deferred one — anyone
repeating this should not expect quantized training to buy headroom on a hybrid
attention model.

## 31. The abstention slice nearly died at the length cap

At the 1024 cap that bf16 requires, abstention fell from 428 samples to **92**
— 4% of the mixture, too thin to teach anything. Diagnosis: the passages are
short (148 tokens median) but the teacher spent ~1,000 tokens deliberating over
whether a short passage contains an answer, so 84% of the traces blew the cap.

Fixed with a **per-domain style**, since §29 established that brevity is not
uniformly safe: shortening the *code* teacher cost 5.4 pp of HumanEval, but
deciding a passage does not mention something needs a look, not an essay. Only
the abstention domain gets the brief instruction; code keeps its long reasoning
and knowledge keeps the 150-word guidance.

Batching now groups by domain so each batch carries one system prompt.

## 32. v3: both predictions failed. v1 stands, and the project should stop here.

| benchmark | v1 | v3 | predicted | outcome |
|---|---:|---:|---:|---|
| MMLU | 83.5% | **83.6%** | ~83.5 | held ✓ |
| TruthfulQA | 79.0% | **77.4%** | ~80.5 | **−1.6, wrong ✗** |
| HumanEval | 90.8% | **87.2%** | ~90 | **−3.6, wrong ✗** |

**The abstention hypothesis (§28) is refuted.** 527 samples whose correct answer
is "the passage does not say" — 19.5% of training, the first data of that kind
this project produced — moved TruthfulQA *down* 1.6 points. The §28 diagnosis
was well-evidenced (11-point hedge gap, p ≈ 0.016) and the intervention still
did not work. That is now four hypotheses about this metric, all wrong.

**The brevity hypothesis (§29) is also refuted, and this one is diagnostic.**
v3 used **v1's exact code traces** — the same 421 samples — and still lost 3.6
points of HumanEval. So the brevity instruction was not what cost v2 its code
score.

| run | samples | code | code share | steps | MMLU | TQA | HumanEval |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 | 1,672 | 421 | 25% | 3,200 | 83.5 | 79.0 | **90.8** |
| v2 | 2,286 | 530 | 23% | 3,200 | 83.1 | 76.9 | 85.4 |
| v3 | 2,704 | 421 | 16% | 5,200 | **83.6** | 77.4 | 87.2 |

What actually tracks the damage is **how much non-code training surrounds the
code data**. v1 has the smallest dataset, the fewest steps, the highest code
share — and the best HumanEval and TruthfulQA by a clear margin. Every attempt
to add data has cost capability outside the slice being added, while MMLU
saturated at v1's level and never moved again (83.5 → 83.1 → 83.6).

### The conclusion this project has earned

Distillation into this student buys **one thing**: it repairs what quantization
broke — reasoning that terminates. That was v1: +5.5 MMLU, +3.0 HumanEval, and
a third off the generation time, from 1,672 samples and three GPU-hours of
training. It is banked and released.

Beyond that, more data does not help, and three rounds of trying have each cost
something. The gap that remains to the 35B teachers is capacity, not recipe, and
a rank-32 LoRA over a few thousand samples cannot close it. **v1 is the release
and the project should stop iterating on data.**

If anything is worth trying later it is qualitatively different — offline
top-k logit distillation, which the shared 248,044-token vocabulary permits and
which trains against the teacher's full distribution rather than one sampled
completion. Not another data mixture.

## 33. The training wall was an implementation gap, and it is now fixed

`mlx_lm.models.gated_delta` trains the gated-delta layers through a **sequential
Python loop over every timestep** — `use_kernel=not self.training`, because the
fast Metal kernel used at inference has no VJP. Its own docstring calls it a
"reference implementation for prompt prefill". Measured on one layer of this
model: **~8 MB of autograd state per token**, scaling linearly, and superlinear
in time.

That is not how the architecture is meant to train. Gated DeltaNet
([arXiv:2412.06464](https://arxiv.org/abs/2412.06464)) inherits the chunkwise
parallel form of DeltaNet ([arXiv:2406.06484](https://arxiv.org/abs/2406.06484)),
which turns O(T) materialised states into O(T/C) and replaces the loop with
matmuls. `src/odistil/gdn_chunkwise.py` implements it.

**One GDN layer, forward + backward:**

| seq | mlx-lm sequential | chunkwise C=64 | gain |
|---:|---:|---:|---|
| 512 | 4.29 GB / 0.36 s | 0.33 GB / 0.04 s | 13x mem, 10x speed |
| 1024 | 10.18 GB / 1.18 s | 0.99 GB / 0.09 s | 10x mem, 13x speed |
| 2048 | 31.44 GB / 4.21 s | 1.74 GB / 0.22 s | **18x mem, 19x speed** |

**Correctness.** Forward and all five gradients agree with the reference to
~3e-7 relative in fp32, including the model's real regime (RMS-normalised
unit-norm keys, weak and strong decay) and the padded-chunk path. On the full
9B, our path agrees with the *inference kernel* at **100% top-1** and differs
from mlx-lm's sequential path by the same margin the kernel does — i.e. it sits
inside mlx-lm's own tolerance. Three regression tests pin this.

**Two things the derivation needed.** `mx.linalg.tri_inv` is CPU-only and has no
VJP, so it cannot appear in a training graph — the unit-triangular inverse is
built from matmuls via 2x2 block recursion in log2(C) levels. And the textbook
formulation divides by the cumulative decay G_j, which underflows and sends the
intermediate writes to infinity on strongly-decaying heads; everything is
therefore expressed in ratios G_j/G_i, bounded by 1 for i <= j.

### What it unlocks

The 1024-token cap was a memory limit, and it was the dominant data filter in
every run: it rejected 46% of everything generated, six times what wrong answers
cost. At 2048, with the same traces already on disk:

| | 1024 cap | 2048 cap |
|---|---:|---:|
| training samples | 2,704 | **4,876** |
| rejected for length | 2,391 | **219** |
| code | 421 | 701 |
| knowledge | 880 | 2,112 |
| truthfulness | 876 | 1,536 |

**+80% training data, no regeneration.** Peak memory at 2048 is 41.8 GB, below
the 47.3 GB that 1024 used to require, and 3072 also fits. Rank, batch size and
full fine-tuning with a factored optimiser are all back on the table.

## 34. Upstream contributions, and one deliberately not made

**PR #1 — chunkwise gated delta (open):**
[ml-explore/mlx-lm#1870](https://github.com/ml-explore/mlx-lm/pull/1870).
`gated_delta_ops` becomes a dispatcher: scalar gating without a mask takes the
chunkwise path, everything else goes to `gated_delta_sequential` (the existing
loop, renamed to say what it is). Five model families train through this code —
`qwen3_5`, `qwen3_next`, `kimi_k3`, `kimi_linear`, `bailing_moe_v3`.

The first attempt conflicted with current `main`, and rebasing was impossible
because syncing the fork needs a `workflow` token scope. Solved by moving the
change into `gated_delta_ops`, a region upstream had not touched, instead of
the dispatch condition, which it had. GitHub now reports MERGEABLE, and the
merged tree was verified locally: suite passes, training dispatch agrees with
the Metal kernel to 1.5e-6.

**PR #2 — fused triangular solve (branch pushed, PR not yet opened):**
`adityak74/mlx-lm:gated-delta-fused-solve`, body in
`docs/mlx-lm-pr2-body.md`. Profiling the chunkwise path showed the per-chunk
inverse was 5.07 ms of ~9.3 ms, and it is used exactly once as `inv @ rhs` — so
the fix is to solve rather than invert. Substitution in one Metal kernel, with
`mx.custom_function` supplying the gradient autodiff cannot derive through a
hand-written kernel:

    x = A^-1 b  =>  b_bar = A^-T x_bar,  A_bar = tril(-b_bar x^T, -1)

Another 2.1-2.7x. Cumulative against stock mlx-lm at 4096 tokens: **108.33 GB /
39.02 s to 4.60 GB / 0.29 s — 24x memory, 135x speed.**

GitHub rate-limited a second PR from the account, which is no loss: stacking on
an unreviewed PR is poor practice, and this one waits for #1.

**#3 — GPU `tri_inv` in mlx core: dropped, deliberately.**

Scoped as a contribution, then abandoned once the ground truth turned up.
[Issue #1392](https://github.com/ml-explore/mlx/issues/1392) is open and
[PR #4375](https://github.com/ml-explore/mlx/pull/4375) already attempted it,
closed as a draft. The reported numbers kill the obvious approach: an
MPS-backed Metal inverse runs **20x slower than CPU at 64x64** and only wins
past 4096x4096, and there is a suspected Apple MPS bug where batched LU returns
zeros for batch 1. The PR is blocked on an unmade maintainer decision, not on
someone writing code.

Our own kernel is the counter-evidence: batched 64x64 unit-triangular solve
holds 1.06 -> 1.22 ms from 32 to 1024 matrices, while CPU goes 0.27 -> 7.18 ms.
A custom kernel handles the regime MPS cannot — though a triangular solve is a
far easier problem than general LU inversion, so it argues a direction rather
than solving their problem. Worth a comment on #1392 if anyone picks this up
again; not worth a duplicate PR.

## 35. v4: the first run without the length bias

Every model so far trained on whatever fit under a 1024-token cap, and that cap
was not a random filter. For v1's recipe:

| | traces | median tokens |
|---|---:|---:|
| trained on | 1,758 | 678 |
| **never seen** | **1,735** | **1,449** |

v1 learned exclusively from the teacher's terse half. §29 had already shown
that shorter code reasoning costs HumanEval — 5.4 points when the code teacher
was told to be brief — so the cap was plausibly doing the same thing silently
to every slice.

The chunkwise path (§33) makes 2048-token training fit, so v4 tests exactly
that one variable:

| held constant | changed |
|---|---|
| 1,639 train / 33 valid, same as v1 | median trace 686 -> **1,010** |
| same slices and 40/25/35 mixture | p90 963 -> **1,758** |
| 3,200 steps, rank 32, top 16 layers, lr 3e-5 | max 1,034 -> 2,058 |

A `max_samples` cap was added to the dataset stage for this: lifting the length
limit otherwise smuggles in extra volume, and volume has failed twice (§32).
The config is built from v1's, not v3's — starting from v3 would have dragged
in the misconception slice and abstention data and confounded the test.

**Training:** 2.79M tokens in 7.1 hours against v1's 1.76M in 8.6 — 58% more
tokens in less wall clock, because the chunkwise path also doubled throughput
end to end (55 -> 111 tokens/s). Train loss 0.119, val 0.305, peak 46.9 GB.

Prediction on the record before measuring: **HumanEval improves most**, since
that is where trace length is known to matter. If nothing moves, four runs will
have failed to beat v1 on data composition, and the remaining lever is method —
offline top-k logit distillation — not another mixture.

## 36. v4 result: the length bias did not matter. Data composition is closed.

| run | MMLU | TruthfulQA | HumanEval | vs v1 |
|---|---:|---:|---:|---|
| stock oQ4 | 78.0% | **80.7%** | 87.8% | −5.5 / +1.7 / −3.0 |
| **v1 (shipped)** | **83.5%** | 79.0% | **90.8%** | — |
| v2 | 83.1% | 76.9% | 85.4% | −0.4 / −2.1 / −5.4 |
| v3 | 83.6% | 77.4% | 87.2% | +0.1 / −1.6 / −3.6 |
| v4 | 83.2% | 76.6% | 89.6% | −0.3 / −2.4 / −1.2 |

**The prediction in §35 was wrong.** HumanEval was supposed to improve most; it
fell 1.2 points. Every v4 delta is inside its noise band, and all three lean
negative — the same shape as v2 and v3.

This was the best-motivated data experiment of the four. The bias was real and
large: v1 trained on 1,758 traces of median 678 tokens and never saw 1,735 of
median 1,449. §29 had independently shown that shortening code reasoning costs
5.4 points. Removing the bias, with sample count and step count held constant,
changed nothing.

### What four runs establish

MMLU is **saturated at ~83.5** and did not move for any intervention: 83.5,
83.1, 83.6, 83.2 across wildly different mixtures, volumes and length
distributions. v1 already matches its own 35B teacher there (83.0).

TruthfulQA is **below stock for every distil** — 79.0, 76.9, 77.4, 76.6 against
80.7 — and four different compositions failed to recover it. Consistent with
§28: the cause is rejection sampling itself, which keeps only confident correct
answers, not any particular slice.

HumanEval tracks nothing we controlled. v1 is best and no variation improved it.

**Data composition is closed.** Four attempts, four failures, including one
aimed at a bias that was measured rather than guessed. Anything further on this
axis should be expected to fail.

### What remains

Offline top-k logit distillation is the only untried change that differs in
kind. Every run so far trained on **one sampled completion per prompt**; logit
KD trains against the teacher's full next-token distribution, which the shared
248,044-token vocabulary makes possible. It needs a custom trainer and roughly
doubles generation cost, and it is off-policy, which the literature favours for
knowledge benchmarks.

Failing that, v1 is the result: +5.5 MMLU and +3.0 HumanEval over the model it
replaces, at the same size and bit width, for a one-time repair of what
quantization broke.

## 37. v5: logit-level distillation

Every run so far minimised cross-entropy against **one sampled completion** per
prompt. The student saw which token the teacher emitted and nothing about how
confident it was or what it nearly said. v5 trains against the teacher's
next-token distribution instead, which the identical 248,044-token vocabulary
across student and both teachers makes possible.

    L = 0.3 * CE(student, sampled token) + 0.7 * KL(teacher || student)

with the KL over the teacher's top-64, renormalised.

**No regeneration needed.** The traces exist, so teacher-forcing them yields
every distribution in one forward pass: 1.3 s per 1,000-token sample against
the ~6 s it took to sample that trace originally. 335 MB of top-k for 1,639
samples; top-64 captures **99.9%** of the teacher's mass.

**Checks before training**, since a silent misalignment here produces a worse
model with no error anywhere: teacher rows equal target tokens exactly, the
actual trace token sits inside the stored top-64 **99.7-99.9%** of the time
(an off-by-one would collapse this), and KL is a real signal rather than noise
— mean KL 0.290 against mean CE 0.466, so at alpha=0.3 the KL term contributes
slightly more than CE.

v5 holds everything else at v1's settings: same 1,639 samples, same slices,
same 3,200 steps, same rank, layers and learning rate. Only the objective
changes. (`max_seq_length` went 1024 to 1280 because v1's samples reach 1,034
tokens under training tokenisation and were being clipped.)

### Two bugs worth recording

Both surfaced as `[metal::malloc] Resource limit (499000) exceeded` around
iteration 700, and the first fix attempt was wrong.

1. **Dense vocabulary tensors.** The loss built two `[B, T, 248320]` float32
   arrays per step — about 2.5 GB at T=1280. Rewritten to derive everything
   from one `[B, T]` log-normaliser plus two gathers, verified **bit-identical**
   (diff 0.00e+00) to the dense form. Peak memory fell 46.1 to 43.5 GB and it
   still died, earlier.
2. **The limit counts live buffers, not bytes.** Which is why reducing memory
   did not help. Every sample has a distinct length, so every step allocated
   uniquely-shaped buffers the allocator could not recycle, and this batcher
   adds two more shapes per step than mlx-lm's — exactly why v1-v4 survived
   3,200 steps through the same loop. Padded widths are now bucketed to
   multiples of 128 and the cache is cleared every 10 steps.

A 1,000-iteration probe cleared both failure points before the full run was
committed to.

**Training:** 3,200 iterations, 1.76M tokens, train loss 0.116, val 0.180,
peak 44.1 GB, ~110 tokens/s. The val loss is not comparable to earlier runs —
different objective — and on this project val loss has not predicted benchmark
outcomes anyway (§32: v2 had the best losses and the worst scores).

## 38. v5 result: logit distillation is worse than stock. Method is closed too.

| run | MMLU | seconds | tokens/question |
|---|---:|---:|---:|
| stock oQ4 | 78.0% | 15,262 | — |
| **v1 (shipped)** | **83.5%** | **13,007** | — |
| v5 | **77.2%** | **69,897** | — |

v5 is the first distil to score **below the stock model it was meant to
improve**, and the run was abandoned after MMLU — TruthfulQA and HumanEval were
never measured, because the wall clock made the remaining two benchmarks a
multi-day proposition for a model already known to have failed.

**The wall clock is the finding, not the accuracy.** 69,897 s against v1's
13,007 s on the same 1,000 questions — 5.4x — at the same size, bit width and
decode settings. v5 did not become confused; it became unable to stop. Under a
2,048-token budget, that converts directly into wrong answers through
truncation, which is the same termination axis as §31's published finding,
driven in the wrong direction by the objective.

The mechanism is consistent with what the objective actually asks for. CE
trains the student toward the single token the teacher committed to. KL trains
it to reproduce the teacher's *entire* distribution at every step — including
the teacher's residual uncertainty about whether to stop reasoning. A 35B
teacher can carry that uncertainty and still terminate; a 9B student at 4.7
bits apparently cannot.

### The result is real, not a bug

A worse model from a silent misalignment looks identical to a worse model from
a bad objective, so three failure modes were checked before accepting it.

1. **Vocabulary identity across teachers.** `logits.py` tokenises with the
   *student's* tokenizer and feeds those ids to whichever teacher owns the
   domain, which is only valid if the vocabularies are index-identical. §37
   verified this for the two Ornith models; the knowledge teacher is
   **Qwen3.6**, which was never checked. Its `tokenizer.json` **does** differ —
   different hash, 18 bytes larger. The vocabularies are nonetheless
   index-identical (248,044 entries, `vocab` and `merges` both equal); the
   difference is the pre-tokenizer regex, which includes `\p{M}` in the letter
   class. That affects how *text* encodes, never what an id means, so stored
   teacher logits are valid in student space.
2. **Top-64 coverage.** §37 asserted 99.9% from the literature rather than from
   these shards. Measured across all 898,322 stored positions: **99.93%**
   (knowledge) and **99.96%** (code); 0.2% of knowledge positions fall below
   0.95 and none below 0.5. Renormalising moves mean top-1 probability from
   0.8849 to 0.8853. The truncated target is faithful.
3. **Span alignment and coverage.** 1,639/1,639 train and 33/33 valid samples
   carry teacher rows, with **zero** target positions uncovered.

The objective was implemented correctly and faithfully, and it made the model
worse. That is a result.

### A latent bug found while checking, fixed

`make_batches` fills rows that have no teacher logits with `ids = 0` and
`lps = -30.0`, and those positions **remain inside the loss mask**. A uniform
`-30` renormalises to a uniform distribution over 64 copies of token id 0 —
i.e. certainty on token 0 — so such rows would train toward garbage, not
"CE alone" as the adjacent print statement claimed. Coverage was complete on
this run so it never fired, but it would silently poison any run whose
extraction was partial. Fixed by masking out uncovered positions explicitly.

### What five runs establish

Data composition is closed (§36). **Method is now closed too.** The remaining
gap to the 35B teachers on MMLU — 83.5 against Qwen's 89.3 — is capacity, not
recipe. Four mixtures and one objective failed to improve on v1; the one that
differed in kind failed worst.

**v1 is the result.** +5.5 MMLU and +3.0 HumanEval over the 4-bit model it
replaces, at identical size and bit width.

## 39. v6: change the filter, not the mixture

§36 closed data composition and §38 closed the objective. What neither touched
is the **filter**. Rejection sampling keeps only confident correct answers, and
§28's per-item control shows exactly what that costs: v1 matches the stock
model on ordinary TruthfulQA questions (81.3% vs 81.4%) and loses 11 points on
questions whose correct answer is "I don't know" (65.0% vs 76.1%, p ~ 0.016).
It is the one deficit on this project with a clean mechanism *and* a clean
control.

v3 already tried the obvious fix and failed: 527 SQuAD-v2 unanswerable items
added as a slice, TruthfulQA down 1.6 points. The likely reason is that it is a
different skill — "the passage does not say" is reading comprehension, while
TruthfulQA hedging is "no good evidence supports this". Adding out-of-domain
abstention data did not transfer, and adding *anything* has now failed four
times.

v6 changes the filter instead. When the knowledge teacher answers an open-ended
question **wrong**, that is by construction a question on which confident
assertion was not warranted — and v1 threw it away. Those prompts go back to
the teacher with its confidence made explicit, and the trace is kept only if it
then declines. The pool is **122 rejected open-ended traces** out of 600.

### Three deliberate constraints

**The target is generated and verified, never fabricated.** The re-ask asks for
a confidence judgement and leaves both outcomes open — it does not instruct the
teacher to decline, which would make the verifier a rubber stamp. A re-ask that
answers confidently is dropped like any other trace that fails its verifier.
This is rejection sampling with the gold inverted.

**Training pairs the abstention with the ORIGINAL prompt**, not the calibration
prompt. The student has to learn to hedge when asked normally, not only when
asked about its confidence.

**MCQ failures are excluded on purpose.** 135 exist and are tempting. MMLU
imposes no penalty for guessing, so teaching abstention on multiple choice
would risk the one metric v1 wins (83.5%, matching its own 35B teacher) to
chase the one it loses. Open-ended only.

### Held constant

Same sources and counts, same 40/25/35 mixture, same 3,200 steps, rank 32, top
16 layers, lr 3e-5, 1024-token cap. `max_samples: 1672` pins the total to v1's
exact count, so recovered abstentions **displace** confident truthfulness
samples rather than adding volume — volume has failed twice (§32), and v4's
null result is only worth anything because it moved one variable. v6 reuses
v1's teacher traces unchanged, so the hedge slice is the only difference.

### Prediction, on the record before measuring

**TruthfulQA improves and MMLU does not move.** Specifically: TruthfulQA 80-81%
(from 79.0%, recovering most of the 1.7-point gap to stock), MMLU within noise
of 83.5%, HumanEval within noise of 90.8% since the code slice is untouched.

The honest odds: the ceiling here is ~1.7 points, the slice is small, and five
runs have failed to beat v1. **If the hedge slice survives verification at
under ~60 samples, any result is inside the noise band and the run answers
nothing** — that is a real possible outcome of this design, not a get-out.

Falsifier: if MMLU drops more than a point, abstention is leaking into multiple
choice despite the exclusion, and the filter change is not separable from the
guessing incentive.

## 40. v6 stopped before training: abstention cannot be distilled from this teacher

v6 was never trained. The hedge stage ran to completion and the slice it
produced cannot support the experiment §39 described.

| stage | count |
|---|---:|
| open-ended traces the verifier rejected | 122 |
| re-asked with confidence made explicit | 121 |
| **teacher answered confidently anyway** | **99** |
| produced no answer within budget | 1 |
| **declined — usable hedge samples** | **22** |
| of those, surviving v1's 1,024-token cap | **0** |

§39 set the threshold in advance: "if the hedge slice survives verification at
under ~60 samples, any result is inside the noise band and the run answers
nothing." It came in at 22 before the length cap and **0 after it**. Training
v6 would have trained on v1's data exactly.

### The interesting number is 99, not 22

**On questions it had already answered wrong, invited explicitly to say it did
not know, Qwen3.6-35B re-asserted a confident answer 99 times out of 121 —
82%.** The re-ask did not instruct it to decline (that would have made the
verifier a rubber stamp); it asked for an honest confidence judgement and left
both outcomes open. Four times in five, the judgement was "I know this", and
it was wrong.

This closes the abstention line for a better reason than v3's failure did.
**The teacher does not have the behaviour we were trying to distill.** No
filter change, mixture, or re-prompt can extract calibrated uncertainty from a
model that does not express it — and sequence-level distillation can only
transmit what the teacher actually emits. §28's diagnosis was right about the
mechanism (rejection sampling keeps only confident answers) and wrong about the
remedy being available at all: the discarded pool is not full of hedges waiting
to be recovered, it is full of confident errors.

### A second reason it would not have worked

The 22 usable traces run **1,295 to 3,158 tokens, median 1,900**. None fit
v1's 1,024-token cap; 14 fit 2,048. The teacher deliberates *at length* before
admitting ignorance — consistent with §31, where asking for shorter abstention
traces produced longer ones. So even the 22 could only be used by also raising
the length cap, which moves a second variable and re-opens the confound v4 was
built to avoid.

### What this run cost and what it is worth

About an hour of teacher generation, no training. It produced a stopping rule
satisfied on the evidence rather than a sixth null result, which is the cheaper
way to close a hypothesis.

**Recorded for whoever revisits this:** the 22 verified hedge traces are kept
at `runs/v6/teacher/hedge.jsonl`. They are real and correctly built. The
supply, not the method, is what failed.

### One bug this run did surface

The first dry run kept 0 of 16 and reported all 16 as "answered confidently".
They were **truncated** — at a 1,024-token budget the teacher spent the whole
budget inside `<think>` and returned an empty answer, and an empty answer is
not a confident one. Fixed to use the teacher's own 2,048 budget with the same
1.5x escalating retry the teach stage uses, and to count no-answer separately.
Had this gone unnoticed the run would have reported a yield of zero for
entirely the wrong reason — §31's truncation finding, reappearing inside our
own pipeline.

### Where the remaining headroom actually is

Five runs varied **data** (v1-v4) and **objective** (v5). None varied the
**adapter**. Every run used rank 32 over the top 16 of 32 layers, chosen once
before v1 and never revisited. §38 concluded the residual gap is "capacity, not
recipe" — but adapter capacity is the one capacity knob the project never
tested. That is the next thing to try, and it is cheap: same data, same
pipeline, one config change.

## 41. v7: vary the adapter -- the axis six runs never touched

v1 through v6 swept data composition four times and the training objective
once. Every one of them used **rank 32 over the top 16 of 32 layers**, chosen
once before v1 and never revisited. §38 concluded the residual gap to the 35B
teachers is "capacity, not recipe" -- but *adapter* capacity is the one
capacity knob nothing measured. v7 tests that conclusion instead of leaning
on it.

Data is v1's, **byte-identical** (`runs/v7/train` is copied from `runs/v1/train`,
sha256 verified), so prompts/teach/dataset are not re-run and cannot drift.
Steps, learning rate, schedule, batch size and length cap are held at v1.

### What fits on 64 GB

| variant | trainable | peak mem | tok/s |
|---|---:|---:|---:|
| v1 (rank 32, top 16) | 43.3M | 47.8 GB | 55 |
| **rank 64, top 16** | **86.557M** | **41.9 GB** | **110** |
| rank 32, all 32 | 86.557M | 55.8 GB | 71 |
| rank 64, all 32 | 173.1M | **OOM** | -- |

Rank 64 across the full stack was the intended experiment and does not fit:
**backprop depth, not rank, is what costs**, since adapting the whole stack
roughly doubles activation memory against v1's top half.

The two survivors are the same size to three decimal places -- rank 32 x 32
layers and rank 64 x 16 layers are both 86.557M -- so they are the same
capacity increase differently allocated, wide over half the stack versus narrow
over all of it. A clean pair, if both are worth running.

**Rank 64 / top 16 runs first**: 41.9 GB against 55.8 leaves real headroom (the
peak above is over 20 iterations and the longest samples may not have appeared
yet; an OOM four hours into an eight-hour run is the expensive failure mode),
and at 110 tok/s it finishes in roughly half the time. Depth is the follow-up
if capacity turns out to matter at all.

One deliberate exception to holding v1 constant: `gdn_chunk: 64` is on, which
v1 predates. It changes speed and memory, not results -- verified to ~3e-7
against the sequential reference (§33) -- and it is what makes 41.9 GB possible.

### Prediction, on the record before measuring

**Nothing moves.** MMLU within noise of 83.5%, TruthfulQA within noise of
79.0%, HumanEval within noise of 90.8%. Six runs have failed to beat v1, MMLU
has not moved for any intervention, and doubling adapter capacity on identical
data is a weaker intervention than several that already failed.

The reason to run it anyway is that "capacity, not recipe" is a conclusion five
runs have leaned on and none tested, and it is cheap -- no teacher generation,
one config change, ~4 hours. A null result converts an assumption into a
measurement. A positive result would reopen the whole project.

Falsifier: if MMLU moves more than a point in either direction, adapter
capacity was a live variable all along and every earlier null result was
measured at the wrong operating point.

### v7 training

3,200 iterations, 1,755,081 tokens, train loss 0.116, **val loss 0.326**, peak
42.5 GB, ~119 tok/s, ~4 hours. Fused, quantized to oQ4 at **4.721 bits with
120/120 promoted modules** (identical to v1's map), installed as
`Ornith-1.5-9B-MLX-distil-v7-oQ4`.

Validation trajectory: 0.723, 0.421, 0.381, 0.387, 0.378, 0.343, 0.398, 0.347,
**0.326**. The bump at iteration 2,400 (val 0.398 while train fell to 0.087)
looked like the overfitting a doubled adapter on a fixed 1,639-sample set would
predict, and it was not: val recovered to 0.347 and finished at its best. Read
as fluctuation, not a trend.

**Loss comparison across runs, with the caveat that it has never predicted a
benchmark here** (§32: v2 had the best losses and the worst scores):

| run | adapter | train | val |
|---|---|---:|---:|
| v2 | rank 32 / top 16 | 0.116 | 0.267 |
| v4 | rank 32 / top 16 | 0.119 | 0.305 |
| **v7** | **rank 64 / top 16** | **0.116** | **0.326** |

v7's val loss is comparable to v2's and v4's — same objective, same metric —
and is *worse* than both. On this project that means approximately nothing.
The benchmark is the measurement.

## 42. v7 result: adapter capacity is not the lever. "Capacity, not recipe" is now measured.

| benchmark | v1 | v7 | delta | items | z |
|---|---:|---:|---:|---:|---:|
| MMLU | 83.5% | 83.2% | −0.3 pp | −3 / 1000 | −0.26 |
| TruthfulQA | 79.0% | **77.6%** | −1.4 pp | −11 / 817 | −0.98 |
| HumanEval | 90.8% | 88.4% | −2.4 pp | −4 / 164 | −1.06 |

(z against a single-proportion SE, which is an upper bound — the paired SE is
tighter, so these are conservative. Per-item files were not collected for v7.)

**The §41 prediction was right: nothing moved.** All three deltas sit inside
one standard error, and the falsifier — MMLU moving more than a point — did not
fire at −0.3. Doubling the adapter from 43.3M to 86.557M trainable parameters,
on byte-identical data with every other hyperparameter held, changed nothing
measurable.

This is the sixth failure to beat v1, and it retires the last cheap hypothesis.
More importantly it converts §38's conclusion from an inference into a
measurement: "capacity, not recipe" was reached by *eliminating* recipes, and
adapter capacity was the obvious confound nobody had tested. It has now been
tested and it is not the lever.

### The shape is the same as every other failure

All three deltas are negative, which is exactly what v2, v3 and v4 did:

| run | MMLU | TruthfulQA | HumanEval |
|---|---:|---:|---:|
| **v1** | **83.5%** | **79.0%** | **90.8%** |
| v2 | −0.4 | −2.1 | −5.4 |
| v3 | +0.1 | −1.6 | −3.6 |
| v4 | −0.3 | −2.4 | −1.2 |
| v7 | −0.3 | −1.4 | −2.4 |

Five runs, twelve of fifteen deltas negative, none positive beyond noise. v1 is
not merely the best result; it appears to be a local optimum that every
perturbation tested — mixture, volume, length distribution, objective, adapter
size — moves away from.

### One thing v7 did preserve

Unlike v5, v7's wall clock is normal: 13,319 s on MMLU against v1's 13,007 s,
2,970 s on HumanEval against 2,785 s. The termination behaviour that §31
identifies as the mechanism behind v1's gains survived intact — v7 is a
healthy model that is simply no better. That is worth recording, because it
isolates v5's failure as specific to the KL objective rather than to any
departure from v1.

### Depth was the other half of this experiment and is not worth running

§41 noted that rank 32 across all 32 layers is **86.557M trainable — the same
size as v7 to three decimals** — and is the only way to reach the bottom half
of the stack, which no run has adapted. It fits at 55.8 GB and 71 tok/s.

It should not be run. v7 establishes that this capacity increase does nothing
when allocated wide; the remaining question is whether the same increase
allocated deep behaves differently, and the prior after six failures is that it
will not. It costs ~7 hours at a tighter memory margin to test a weaker version
of a hypothesis just falsified. Recorded as deliberately declined, with the
config recipe in §41 if anyone disagrees.

### Status

**v1 remains the release.** Seven attempts, one success, and the failures now
cover data composition (4), training objective (1), filter (1, stopped on
evidence) and adapter capacity (1). The remaining gap to the 35B teachers is
the student's own capacity at 9B and 4.72 bits, which no amount of distillation
recipe reaches.

## 43. Correction to §42, and v8: error-conditioned data

§42 said "every axis available to a distillation recipe is closed". A second
reviewer read the repo and pushed back, correctly: what is closed is **offline,
teacher-generated, sequence-level KD under LoRA**, plus one forward-KL run. It
was too strong, and this section replaces it.

What v1-v7 never tested, in the reviewer's list and my own reading of it:

- **on-policy / student-error-conditioned data** -- every pool was static
- **skew or adaptive KL** in place of the forward KL that v5 used
- **termination-aware KD** -- masking stop tokens from the distribution loss
- **preference distillation** between verified and failed trajectories
- **quantization-aware KD** -- training through the oQ4 quantizer
- **full-parameter updates** -- v7 varied LoRA rank, not the update subspace

Each is a different kind of thing from a data mixture. Taking them in order of
evidence-per-hour, with one variable moved per run.

### v8: train on what the student gets wrong

Every earlier training set was a function of the *pool's* composition. v8 makes
it a function of the *student's errors*:

    fresh pool -> v1 rolls out -> verify -> keep the failures
               -> teacher answers those -> verify -> sequence CE
               -> continue from v1's fused bf16

Most of a static pool re-teaches what the student already knows. A pool of v1's
failures does not. This is DAgger's insight at the prompt level; the token-level
form (arXiv:2306.13649) is the natural next step once this one is measured.

**Held at v1:** objective (sequence CE, the one known to work), teachers,
verifiers, rank 32 / top 16, lr 3e-5, cosine. **Changed with a reason:**
`max_seq_len` 2048 -- the failures are hard prompts with long teacher traces,
and the chunkwise path makes 2048 fit; and `train.base = runs/v1/fused`, so
the adapter continues from the best checkpoint rather than restarting from the
stock base. Both are shared with the control arm, so neither confounds the
comparison that matters.

**The control arm is not optional.** `v8-control` trains on a *random* subset
of the same pool, same size as the failure arm, everything else identical.
Without it a gain is confounded with "more training"; with it, the only
difference between arms is *which* prompts. v2 and v3 showed more data hurts,
so a positive result would already be notable -- but the control makes it a
measurement rather than an argument.

### Supply

MBPP is fully consumed (all 974 items across every split went into v1), so the
fresh code slice comes from **KodCode-V1** (484k items, pytest-style tests).
The tests import from a `solution` module; the normaliser drops that line --
the model's own code defines the function in the same program -- and appends a
runner that calls each test function, so they execute through the existing
sandbox unchanged. Rows the dataset itself flags, and rows with benchmark
similarity >= 0.95, are skipped, on top of the 13-gram decontamination the
dataset stage always runs.

The pool is 5,000 prompts (2,500 knowledge MCQ, 1,200 truthfulness, 1,300
code), sampled under a different seed with every v1 id explicitly excluded.

### Cost

Student rollouts dominate: ~13 s/item with thinking on, so ~18 hours for the
pool, unattended. Teacher generation runs only on failures (~2 h). Training on
~600 samples is ~2 h per arm. The rollout is the price of knowing what the
student does not know; everything after it is cheap.

### Prediction, on the record before measuring

**v8 beats v8-control on at least one benchmark by more than noise, and
v8-control does not beat v1.** The second half is the v2/v3 pattern -- more
generic data does not help -- and the first half is the actual hypothesis. If
both arms sit inside v1's noise band, error-conditioning at the prompt level is
not enough and the token-level (GKD) form is the next thing to test. If
v8-control beats v1, continuing from v1's checkpoint is doing the work, not the
prompt selection, and that is worth knowing too.

Falsifier: v8 below v1 on HumanEval. The failure prompts are, by construction,
the hardest ones, and long teacher traces on hard code are the same input that
made v4's code slice not help. If that pattern repeats here it is a real
limit on what hard-prompt data can do for this student.

### v8 rollout and yield (recorded as measured)

**v1's accuracy on the fresh 4,974-prompt pool**, rolled out at oQ4 with the
eval budgets:

| kind | n | correct | wrong | truncated |
|---|---:|---:|---:|---:|
| code (KodCode) | 1,300 | **49.2%** | 661 | **461** |
| MCQ | 2,878 | 91.2% | 252 | 54 |
| open-ended QA | 796 | 58.0% | 334 | 85 |

KodCode is far harder than MBPP for this student, and 70% of its code failures
are truncations, not wrong code. MCQ at 91% is roughly v1's MMLU, so its 252
failures are the genuinely hard tail. Failure arm: **1,247 prompts**; control:
1,247 random.

**Teacher yield on the failure arm was low, and the reason is length:**

| rejection | n |
|---|---:|
| teacher produced no answer within 4,608 tokens | 267 |
| trace over the 2,048-token cap | 182 |
| code failed tests | 218 |
| MCQ / QA wrong against gold | 250 |
| **kept** | **330 (26%)** |

On the code slice specifically the 35B teacher verified on only 29% of
attempts and **failed to terminate on 38%** even after two budget escalations.
That is the termination finding (docs/TERMINATION.md) appearing in a 35B model
on prompts selected by a 9B's failures, and it goes in that write-up.

**Two decisions made here, both shared by the control arm so neither confounds
the comparison:**

- The 2,048 cap stays. Raising it to 3,072 would recover ~182 traces but is a
  memory risk on an unattended run, v4 showed the length axis alone does
  nothing, and it is a second variable.
- The control is pinned to **330 samples and 660 steps**, matching the failure
  arm exactly. Same N, same steps; only the prompts differ.

**The mixture is not held.** The failure arm came out 58% truthfulness / 27%
knowledge / 15% code, because composition is downstream of which prompts fail.
That is inherent to the design, applies identically to both arms' logic, and
is one more reason the control matters.

**A bug found on the way:** both `teach` and `rollout` filtered each batch to
its first item's domain and then advanced by `batch_size`, dropping the rest of
any chunk that straddled a boundary. `teach` has done this silently since v1
(≤ batch_size−1 prompts per boundary per run). Both now group first and batch
within groups.

### v8 and v8-control: trained and installed

Both arms: 330 samples (324 train / 6 valid), 660 steps, rank 32 / top 16
(43.278M trainable, v1's exact adapter), lr 3e-5 cosine, 2,048 cap, continued
from v1's fused bf16. Peak 46.9 GB, ~110-120 tok/s, ~1.5 h each.

| arm | tokens | train loss | val loss (6 items) |
|---|---:|---:|---:|
| v8 (failures) | 616,937 | 0.079 | 0.281 → 0.285 |
| v8-control (random) | 579,319 | 0.064 | 0.368 → 0.358 |

Val is over six items and is not comparable across arms (different items).
Both quantized at 4.721 bits, 120/120 promoted, installed as
`Ornith-1.5-9B-MLX-distil-v8-oQ4` and `Ornith-1.5-9B-MLX-distil-v8ctl-oQ4`.

The comparison that matters is three-way: **v8 vs v8-control** (does prompt
selection matter?) and **each vs v1** (does continuing from v1 on 330 more
samples do anything at all?). §43's prediction stands as written.

## 44. v8 result: both arms below v1. The second pass is the damage, not the prompts.

| | v1 | v8-control | v8 | ctl − v1 | v8 − ctl |
|---|---:|---:|---:|---:|---:|
| MMLU | **83.5%** | 83.2% | 81.6% | −0.3 (z −0.3) | −1.6 (z −1.4) |
| TruthfulQA | **79.0%** | 77.7% | 76.3% | −1.3 (z −0.9) | −1.4 (z −1.0) |
| HumanEval | **90.8%** | 86.6% | 85.4% | −4.2 (z −1.9) | −1.2 (z −0.5) |

**The §43 prediction was wrong on both halves.** v8 does not beat the control
on any benchmark — it is *below* it on all three, each inside noise. And the
control does not sit at v1: it is below v1 on all three, with HumanEval at
−4.2. The pre-registered falsifier (v8 below v1 on HumanEval) fired at −5.4.

### What the control isolates

The two arms differ only in which 330 prompts. Both lost to v1 by similar
amounts, so **continuing from v1's checkpoint with a second LoRA pass on 330
more verified samples degrades all three benchmarks regardless of what the
samples are.** Prompt-level error-conditioning did not help; if anything it
cost another point, inside noise. The damage is the second pass.

That is the sharpest statement yet of something every run since v2 has hinted
at: **v1 is a sharp optimum.** Nine of nine comparable deltas across v8 and
its control are negative, joining v2/v3/v4/v7's twelve of fifteen.

### The wall-clock split is real and unexplained

| | MMLU | TruthfulQA | HumanEval |
|---|---:|---:|---:|
| v8 vs v1 | +6% | +12% | +12% |
| **v8-control vs v1** | **+46%** | **+80%** | **+75%** |

The control — random prompts — became dramatically slower to answer; the
failure arm barely did. Same recipe, same base, same steps, same cap; only the
prompts differ. One candidate: the failure arm's traces are, by selection,
the ones a 35B *did* finish on hard problems and that fit under 2,048 tokens —
a sample biased toward clean termination — while the random arm's traces are
not. It is the one thing error-conditioning visibly did. It is also possible
the oMLX server was loaded during the control run; a +46–80% swing is larger
than any previous run-to-run noise, but that cannot be ruled out from here.
Recorded, not claimed.

### What eight runs now establish

| axis | runs | outcome |
|---|---:|---|
| data composition | v2 v3 v4 | fails |
| training objective (forward KL) | v5 | fails worst |
| the filter | v6 | no supply |
| adapter capacity | v7 | no effect |
| **error-conditioned prompts, continued from v1** | **v8 + control** | **fails; the continuation itself costs** |

Prompt-level on-policy selection is closed. The reviewer's list (§43) still
has untried items — token-level GKD with skew KL and termination masking,
preference KD, quantization-aware self-distillation, full-parameter updates —
but v8 adds a constraint on all of them: **any method that starts from v1 and
trains further has to first beat the ~1–4 point tax that a second pass appears
to impose**, or it has to train from the stock base and re-earn v1's gains from
scratch. Neither is cheap, and the prior after eight runs is poor.

### One finding from the yield, worth keeping regardless

On prompts a 9B fails, the 35B code teacher verified on 29% and **did not
terminate on 38%** even at 4,608 tokens — against 73% verified on random
prompts from the same pool. The termination failure is not specific to the
quantized 9B. Added to docs/TERMINATION.md.

## 45. Budget forcing: the termination finding, tested at the harness

If the 4-bit student's dominant failure is *stopping* (§31, docs/TERMINATION.md),
then the cheapest possible test is to stop it for it. Budget forcing
(Muennighoff et al., s1, arXiv:2501.19393): when the reasoning block has not
closed by the time the budget is nearly spent, close it and let the model
answer with what it has. No training; a harness change.

Implemented as two phases in `generate_batch`: `max_tokens − N`, then N more
for anything that hit the cap. Items still inside `<think>` get the block
closed for them (FORCE_STR) before continuing; items that had closed it and
were mid-answer simply continue. No item receives more than `max_tokens` in
total. `odistil eval --force-budget N`.

**The first version had a real flaw**, caught by the per-item diff: the
reserve was taken out of *every* item's budget, so one item that had finished
reasoning and was writing its code was cut at phase 1 and scored wrong. The
per-item file is what surfaced it; the aggregate (+1) looked fine.

### Smoke: 24 seeded HumanEval items, v1 at oQ4, 2,048-token budget

| | correct | truncated | forced | forced → correct |
|---|---:|---:|---:|---:|
| plain | 19 / 24 | 2 | — | — |
| **forced, N=256** | **21 / 24** | 0 | 3 | 2 |

Both truncations were forced; one recovered. The third forced item had *not*
truncated under plain decoding — it reasoned to ~2,050 tokens and answered
wrong — and cut at 1,792 it committed to the right answer. n=1, but it is the
shape the finding predicts: past some point, more reasoning is not helping.

Zero regressions after the fix. Wall clock unchanged (253 s vs 251 s).

### What this does and does not show

It shows the mechanism is real and recoverable at the harness: on a 24-item
sample, forcing turned two wrong answers into right ones and cost nothing. It
does not yet show the size of the effect at the deployment budget of 4,096, or
on MMLU where truncated multiple-choice answers are already partly recoverable,
or whether the gain holds at n=164. That is the next measurement, and it needs
no training: full HumanEval plain vs forced at 2,048 and at 4,096.

## 46. v9: self-distillation of the HALT traces

§45 and the paper showed the harness fix is worth +10.4 HumanEval on stock oQ4
and +7.3 on v1, with no training. v9 asks whether that can be moved into the
weights: roll the student under HALT, keep every verified-correct trace, train
on it. The traces the harness closed are the interesting ones — they end early
and still verify, which is the behaviour quantization removed (§45, paper §5).

### Why this loop and not the others considered

- On-policy self-generated data has no distribution shift and no teacher
  swap. v8's teacher failed to terminate on 38% of the student's code
  failures (§43); the student's own correct traces cannot have that problem.
- Synthetic prompt generation was set aside: the v8 pool (5,000 prompts,
  disjoint from v1) is unused as training data, and teacher-written MCQ has
  no verifier, which the rejection-sampling invariant forbids.
- Preference training on (short correct, long truncated) pairs is the natural
  follow-up but needs a DPO trainer on MLX that does not exist here yet.

### Design

- **Rollout:** v1 at oQ4 over runs/v8pool, `max_tokens 1536`, `force_budget
  256` — 1,280 tokens of reasoning, then `</think>` forced for anything still
  inside the block. The paper's 2,048 + 256 setting would put most forced rows
  over the 2,048 training cap (prompt + trace) and the dataset stage would
  reject them as too_long; 1,536 keeps them.
- **Harvest** (`rollout.mode: harvest`, new): every correct roll becomes a
  teacher-shaped row in `teacher/self.jsonl`, tagged `forced`. The harness
  sentence ("I am out of thinking budget…") is stripped; the row closes
  `</think>` where the harness closed it, so the model learns to stop, not to
  say it was stopped.
- **Mix:** self rows + v1's teacher traces (copied into `teacher/`), same
  40/25/35 domain mix, `max_seq_len 2048` with the chunkwise path.
- **Base:** the stock bf16 student, one pass, v1's rank/layers/lr. Not v1's
  fused checkpoint — six second-pass runs on it went negative on 21 of 24
  deltas (§44). This means the v1 half is re-filtered at 2,048 rather than
  1,024; v4 is the nearest reference for that (length-unbiased, ≈ v1).
- **Control** (`configs/v9-control.yaml`): identical, forced rows dropped,
  same rollouts. The one variable is whether the forced traces are in the mix.

### Go/no-go, set before the rollout started

1. After harvest: fewer than ~150 forced-correct code rows means too little
   forced data to learn from; do not train.
2. After training, in our harness at 2,048 on HumanEval, plain decoding:
   v1 = 130/164, 24 truncated (HALT: 142). v9 needs ≥ 136 and a clearly
   lower truncation count than v1. If truncation does not move, stop after one generation and
   report it beside v8.
3. If it moves: roll v9 under HALT, retrain as v10, stop when truncation
   stops falling.

Smoke (16 pool prompts): 14/16 correct, 1 forced (wrong), harvest wrote 14
rows and copied both v1 teacher files. Full rollout started 2026-09-17.

## 47. v9 rollout at 1,536: the yield check fails, and the budget is why

Full pool, v1 at oQ4, 1,280 tokens of reasoning + 256 reserve (2026-09-17/18):

| kind | n | correct | forced | forced ∧ correct |
|---|---:|---:|---:|---:|
| mcq | 2,878 | 91.8% | 152 | 71 |
| qa | 796 | 59.8% | 137 | 41 |
| code | 1,300 | **20.9%** | 911 | **27** |

Criterion 1 in §46 asked for ~150 forced-correct code rows; there are 27.
That is a no-go on the criterion as written, but the criterion assumed the
reasoning budget was adequate and it was not: the same student solved 49.2%
of this pool's code at 4,096 (§43) and 20.9% here, with 911 of 1,300 items
hitting the cap. HALT recovers items that reasoned to a conclusion and failed
to stop; at 1,280 tokens most code items have not reached one. The budget was
set to keep prompt + trace under the 2,048 training cap, which was the wrong
constraint to protect.

MCQ and QA behave as the paper predicts: forcing recovers about half of the
forced MCQ items and a third of the forced QA items.

**Action:** code re-rolled at the paper's setting, 2,048 + 256, with
`max_seq_len` and `max_seq_length` raised to 3,072 (chunkwise path; v7 fit
2,048 at 41.9 GB). The 1,536 code rows are kept at
`runs/v9/rollouts-code-1536.jsonl`. MCQ/QA rows stand. Criterion 1 is
re-applied to the 2,304 code roll; under 150 there is the negative.

## 48. v9 result: negative at the data stage. Stopped before training.

Code re-rolled at 2,304 (2,048 + 256), stopped at 1,016 of 1,300 once the
answer was clear (2026-09-18):

| code budget | n | correct | forced | forced ∧ correct |
|---|---:|---:|---:|---:|
| 1,536 | 1,300 | 20.9% | 911 | 27 |
| 2,304 | 1,016 | 33.6% | 551 | **7** |

Raising the budget fixed accuracy and *lowered* the forced-correct yield.
Projected to the full pool that is ~9 rows against the 150 the design
required (§46). v9's hypothesis — that HALT-closed code traces exist in
enough quantity to distill the stopping behaviour into the weights — is
false on this pool at this model. There is nothing to train on.

Why the paper's +12 on HumanEval does not carry over: forcing recovers items
that had reached an answer and failed to stop. On HumanEval at 2,048 that
was 12 of v1's 34 truncations. On KodCode, harder by construction, the items
still reasoning at 2,048 have not reached one; forced, 544 of 551 fail the
tests. HALT is a decode-time gain on the benchmark distribution and does not
manufacture training data on a harder one.

What the rollout did yield: 3,390 verified-correct MCQ/QA self-traces at
1,536 (112 of them forced) and ~340 code at 2,304. A run on those would test
"does on-policy self-distillation help at all", which v8-control already
answered (no second pass has beaten v1, §44), or "does early stopping
transfer from MCQ/QA to code", a new hypothesis with a weak prior. Neither
is v9. Not run.

Cost: ~15 h of decode across the two rolls. No model produced. Rollouts kept
at runs/v9/rollouts.jsonl and rollouts-code-1536.jsonl.

## 49. v10: preference training on termination pairs. Negative.

v9 (§46–48) failed for lack of positive examples. v10 used the abundant
negative one instead: the student's own trace that reasoned past the budget.

### Design

- **Pairs** (`odistil pairs`, new): chosen = a teacher trace that verified
  (runs/v8/teacher), rejected = v1@oQ4's failed trace on the same prompt
  (runs/v9 rollouts, the harness sentence stripped and the block left
  unclosed). 378 pairs at a 3,072 cap: code 99, mcq 107, qa 172; 159 rejected
  traces unterminated.
- **Loss** (`odistil pref-train`, new): sigmoid DPO, β 0.1, reference = v1
  fused, plus 0.2 × CE on the chosen trace (RPO). LoRA rank 32 / top-16 over
  v1 fused, lr 1e-5 cosine, 720 steps (2 epochs), dropout 0, grad clip 1.0.
- **Engineering:** mlx-lm has no preference trainer. mlx-lm-lora
  (Goekdeniz-Guelmez) has one, but it scores both sequences and a resident
  reference model in one graph with dense float32 cross-entropy, none of
  which fits a 9B at 3,072 on 64 GB. Ours factorises the DPO gradient
  (∇L = −β σ(−Δ)(∇log π_c − ∇log π_r)) into one no-grad forward and two
  single-sequence backward passes, with the reference precomputed once and
  cached. Peak 34.8 GB on the longest pair. The gated-delta layer must stay
  in train mode throughout: its eval-mode Metal kernel has no vjp and differs
  from the chunkwise path by ~0.003 nat/token, which biased Δ at init by
  0.24 until reference and policy shared one path.

### Training

Loss reached 0.00 by step 50 and the train margin grew to +25–30 by step
400; validation margin +9 to +13 from step 150 on, accuracy 1.00 throughout.
The pairs are trivially separable — a teacher trace against an unterminated
student trace — so DPO learns them at once and then only widens the margin.

### Result (our harness, HumanEval, 2,048 plain, oQ4)

| | correct | truncated |
|---|---:|---:|
| v1 | 130 | 24 |
| v10 step 720 | 123 | 30 |
| v10 step 150 | 116 | 39 |

Per item against v1, step 720: 12 correct→truncated, 8 correct→wrong, 6
truncated→correct, 7 wrong→correct. Mean output length unchanged (1,139 →
1,112 tokens). Step 150: 17 correct→truncated, 7 truncated→correct. Both
checkpoints truncate *more* than v1 (30 and 39 against 24). This is churn
plus a drift toward longer traces, not a shift toward stopping; the early
checkpoint is worse, so it is not over-optimisation either.
Both fail criterion 2 of §46. No oMLX run.

### Reading

Sequence-level DPO on (terminated, unterminated) pairs moves the sum of
log-probs over whole traces; the behaviour we want is a decision at one
position, the `</think>` token when the reasoning has reached an answer.
Nothing in the pair localises the credit there, so the adapter buys margin
by perturbing the whole trace, and at oQ4 that perturbation lands as noise.
A fix would need token-level credit (a termination-position loss, or
on-policy RL with the verifier as reward and length as cost), which is a
different experiment. Preference training as tried here is closed.

Cost: ~6 h training + 2 × 1.5 h fuse/quantize/eval. Adapters and per-item
records kept under runs/v10 and runs/v10-s150.

## 50. Small probes, decode-time: a soft HALT beats the hard one

After v9 and v10 (§46–49) the plan changed to many small probes, 30 minutes
each, all on v1 @ oQ4, HumanEval @ 2,048 in our harness, plain baseline
130 / 24 truncated, hard HALT (N = 256) 142 / 0.

First, the GRPO feasibility check (`scripts/grpo_signal.py`): 8 samples at
T = 0.8 per KodCode prompt at 2,048. Stopped at 27 groups: 22% had both a
correct and a truncated member, 59% had no correct member at all. Below the
30% bar set beforehand; a GRPO run on this pool would spend most of its
generation on zero-gradient groups. Not pursued. QLoRA against the oQ4
forward was also closed for deployment reasons: oMLX's `model_discovery.py`
skips any directory with `adapter_config.json` ("oMLX does not support
LoRA/PEFT adapters"), so a 4-bit-trained adapter would have to be fused and
re-quantized, which discards most of the delta (§30 already ruled it out on
speed).

Then five decode-time probes, no training, same weights (`eval --rep-penalty
/ --temp / --think-bias START:SLOPE`; records now keep the tail of each
truncated think block):

| probe | correct | truncated |
|---|---:|---:|
| plain | 130 | 24 |
| repetition penalty 1.1 | 128 | 24 |
| temperature 0.6 | 122 | 28 |
| `</think>` bias +0.005 / token past 1,024 | 130 | 24 |
| **`</think>` bias +0.02 / token past 1,024** | **147** | **0** |

Of the 24 truncated traces: 2 loop, 14 had already written code inside the
think block and kept verifying it, 8 were still reasoning. Truncation is not
degeneration, which is why penalty and sampling do nothing.

The ramp adds `slope × (n − start)` to the `</think>` logit while the block
is open, nothing before `start`, nothing after the block closes. It is HALT
without the cliff: instead of a forced close at one position, the stop
decision gets gradually cheaper from 1,024 on, and the model chooses where.

Per item against plain, slope 0.02: 130 C→C (**no regressions**), 12 T→C,
12 T→W, 5 W→C, 5 W→W; all 76 items that finished under 1,024 tokens are
byte-identical. Mean output 1,139 → 1,085 tokens. 147 is +17 over plain, +5
over hard HALT, and +2 over the bf16 parent at 4,096 (145). The five W→C
items reasoned past 1,024 tokens to a wrong answer under plain decoding and
answer correctly when nudged to stop sooner — the paper's smoke-test
observation (§45, n = 1) at n = 5.

Next, same unit size: a sweep over (start, slope) and the stock oQ4 build,
then MMLU on a subset, then the write-up as a v2 of the paper.

## 51. The ramp is a plateau, and the stock build gains 31 items

Sweep on v1 @ oQ4, HumanEval @ 2,048, per-item against plain (130 / 24):

| start / slope | correct | truncated | regressions | mean tokens |
|---|---:|---:|---:|---:|
| 1,024 / 0.01 | 133 | 17 | 0 | 1,135 |
| 1,024 / 0.02 | 147 | 0 | 0 | 1,085 |
| 1,024 / 0.04 | 147 | 0 | 0 | 1,016 |
| 768 / 0.02 | 146 | 0 | 1 | 1,032 |
| 1,280 / 0.02 | 144 | 0 | 0 | 1,126 |

A plateau from slope 0.02 up and start 768–1,280; only 0.01 is too weak to
close the block by 2,048. No tuning needed; 1,024 / 0.02 is the setting.

Stock oQ4 (no distillation), same budget: plain **108 / 48 truncated**, ramp
1,024 / 0.02 **139 / 0**, zero regressions, 28 T→C and 3 W→C. +31 items,
+18.9 points. For scale, the paper's stock-with-HALT at 4,096 was 135 and
v1 plain at 4,096 was 138: the ramped stock build at half the budget beats
both. Untrained weights, one logits processor.

Reading: quantization did not remove the model's ability to stop, it
lowered the `</think>` logit relative to "keep going" at the positions
where a decision is due. A bias that grows with position restores the
decision without choosing the position for the model. HALT at a cliff
was the coarse version of the same fix.

Next: MMLU-250 @ 3,072 with the ramp from 2,048, then the paper v2.

## 52. MMLU confirms; the probe series is closed

v1 @ oQ4, MMLU-250 @ 3,072, ramp from 2,048 at 0.02: **210 / 0** against
plain 206 / 10 and hard HALT 209. 206 C→C, 4 T→C, 6 T→W, no regressions.
Smaller than HumanEval because MMLU truncates 4% of items, not 15%; same
shape. Three benchmarks × builds, zero regressions in every one.

Series summary (all decode-time, same weights): repetition penalty and
sampling do nothing; a hard limit recovers truncations; a soft ramp
recovers the same truncations plus a handful of wrong-by-overthinking items,
with no cliff to place. It supersedes hard HALT as the recommended fix and
goes into the paper as v2, with v9/v10 as the two training-side negatives
that bound it.
