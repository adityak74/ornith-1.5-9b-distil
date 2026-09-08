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
