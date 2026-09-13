# Repairing what quantization broke: five attempts at distilling Ornith-1.5-9B

A complete record of the project — five distilled models and a sixth stopped
before training, one shipped, one upstream PR open with a second branch behind
it, one finding worth publishing, and a larger pile of wrong hypotheses than
right ones.

Detail for every decision is in [`DECISIONS.md`](DECISIONS.md) (40 sections);
this is the paper-shaped summary. Written for whoever picks this up next,
including future me.

---

## Abstract

A 9B reasoning model quantized to 4.7 bits loses 4.6 points of MMLU and 3.7 of
HumanEval against its bf16 parent. We asked whether sequence-level distillation
from two 35B teachers could repair that loss without adding bits, and found
that it can: **v1 recovers +5.5 MMLU and +3.0 HumanEval over the 4-bit model it
replaces, at identical size and bit width**, beating the 1.8x larger 8-bit
build on both and matching its own 35B teacher on MMLU.

Five subsequent attempts to improve on v1 — three data-composition changes, one
change of training objective, and one change of the *filter* — **all failed**,
and the objective change failed worst. The negative results are the more useful
half of this document, because they localise where the remaining headroom is
not.

Along the way the dominant engineering constraint turned out to be a sequential
loop in a dependency rather than the hardware; removing it yielded **24x memory
and 135x speed** on the gated-delta training path and is now upstream.

The finding we think generalises: **quantization damages a reasoning model's
ability to stop more than its ability to reason**, and any benchmark of a
reasoning model that does not report its truncation rate is partly reporting
its own token budget.

---

## 1. Setup

**Student.** Ornith-1.5-9B, a hybrid-attention reasoning model with gated
DeltaNet (GDN) linear-attention layers, in the `oQ4` quantization: 4-bit affine
at group size 64 with 120 modules promoted to 5/6/8 bits, averaging 4.72 bits.
The promotion map is recoverable from the checkpoint's `config.json` and was
replayed exactly, so every comparison is like-for-like rather than "our q4 vs
their oQ4".

**Teachers, split by measurement rather than by reputation.**

| | MMLU | TruthfulQA | HumanEval | sampling speed |
|---|---:|---:|---:|---|
| Qwen3.6-35B-A3B-4bit | **89.3%** | **89.2%** | 93.3% | 1.0x |
| Ornith-1.5-35B-A3B-4bit | 83.0% | 86.4% | 93.3% | **4.6x** |

Qwen is 6.3 points better on MMLU, so it teaches knowledge and truthfulness.
Ornith-35B ties it on code while sampling 4.6x faster, so it teaches code. All
three models share an **index-identical 248,044-token vocabulary** (verified:
`vocab` and `merges` equal across all three, though Qwen's pre-tokenizer regex
differs), which is what later made logit-level distillation possible at all.

**Method.** LoRA (rank 32, top 16 of 32 layers, lr 3e-5) over the **bf16** base,
then fuse, then quantize — so the quantizer starts from an improved checkpoint
instead of compounding two lossy steps. Every teacher trace is **rejection
sampled**: gold letter for multiple choice, gold aliases for open questions,
and generated code actually executed against the source dataset's tests.

**Hardware.** One Apple M4 Pro, 64 GB unified. A 9B full fine-tune does not
fit; LoRA over bf16 does, with one model resident at a time.

**Protocol.** MMLU 1,000-question seeded sample of 14,042; TruthfulQA all 817;
HumanEval all 164; thinking enabled; greedy. Every number below was measured on
the same oMLX server under the same protocol.

---

## 2. Results

| model | size | MMLU | TruthfulQA | HumanEval |
|---|---:|---:|---:|---:|
| Ornith-1.5-9B-oQ4 (stock) | 4.9 GB | 78.0% | **80.7%** | 87.8% |
| Ornith-1.5-9B-oQ8 | 8.9 GB | 83.1% | 80.7% | 88.4% |
| Ornith-1.5-9B bf16 | 17 GB | 82.6% | 80.5% | **91.5%** |
| **distil-oQ4 — v1, shipped** | **4.9 GB** | **83.5%** | 79.0% | 90.8% |
| distil-v2-oQ4 | 4.9 GB | 83.1% | 76.9% | 85.4% |
| distil-v3-oQ4 | 4.9 GB | 83.6% | 77.4% | 87.2% |
| distil-v4-oQ4 | 4.9 GB | 83.2% | 76.6% | 89.6% |
| distil-v5-oQ4 | 4.9 GB | 77.2% | not run | not run |
| Ornith-1.5-35B-A3B-4bit (teacher) | 18 GB | 83.0% | 86.4% | 93.3% |
| Qwen3.6-35B-A3B-4bit (teacher) | 19 GB | **89.3%** | **89.2%** | 93.3% |

**v1 shipped**: +5.5 MMLU and +3.0 HumanEval over the 4-bit model it replaces,
at identical size and bit width. It beats the 1.8x larger oQ8 on both, edges
out the 3.5x larger bf16 base on MMLU, and matches its own 35B teacher on MMLU.
It is also 33% faster on HumanEval and 15% on MMLU.

**v2 through v5 were not shipped.** None beat v1; v5 did not beat the stock
model. That is most of this document.

### What each run changed

| run | one-line change | MMLU | TruthfulQA | HumanEval |
|---|---|---:|---:|---:|
| v1 | baseline recipe | **83.5%** | 79.0% | **90.8%** |
| v2 | misconception slice replaces TriviaQA; brief code | 83.1% | 76.9% | 85.4% |
| v3 | abstention slice added; v1's code traces restored | 83.6% | 77.4% | 87.2% |
| v4 | 1024-token length bias removed, all else held | 83.2% | 76.6% | 89.6% |
| v5 | **objective changed**: CE → CE + KL on teacher top-64 | 77.2% | — | — |
| v6 | **filter changed**: recover verified abstentions | stopped before training |  |  |

---

## 3. What worked

**Two teachers, routed by measurement.** Cheaper than using the stronger
teacher for everything, and the MMLU gain came through intact.

**Rejection sampling.** Qwen is 89.3% on MMLU, so roughly one unfiltered
knowledge trace in nine would have been a confidently argued wrong answer.
Filtering them out is most of why v1 works. It also costs TruthfulQA — see §5.
Both are true, and the trade was worth making.

**Quantizing after training, not before.** See §1.

**Reproducing the quantization map exactly.** Without this the headline
comparison would have been measuring our quantizer against theirs.

**The chunkwise gated-delta training path.** The largest engineering result of
the project, and it came from asking why a constraint existed rather than
designing around it. See §6.

---

## 4. The finding worth publishing

**Quantization damages a reasoning model's ability to stop more than its
ability to reason.**

On a strict harness the stock 4-bit model **never reaches an answer on 24% of
HumanEval problems**. v1 cuts that to 12%, and **22 of the 28 problems it gains
are ones where the stock model ran out of budget mid-reasoning**. Mean
generation drops from 1,721 to 1,414 tokens.

It is not a scoring artefact: it reproduces as a **33% wall-clock reduction on
a harness that does not penalise truncation at all**. And the same weights
score 82.3% or 88.4% on HumanEval depending only on whether the budget is 2,048
or 4,096 tokens — so **any benchmark of a reasoning model that does not report
its truncation rate is partly reporting its own token budget.**

Distillation on traces that terminate cleanly repairs most of it. That is what
v1 is. v5 is the same axis pushed the other way (§5.3), which is indirect
confirmation of the mechanism.

---

## 5. What didn't work

### 5.1 Four data experiments, four failures

Volume was not the answer: v1 used 1,672 samples, v2 2,286, v3 4,075, and both
larger runs lost ground.

**MMLU is saturated at ~83.5** — 83.5, 83.1, 83.6, 83.2 across wildly different
mixtures, volumes and length distributions, with v1 already matching its own
35B teacher (83.0). **TruthfulQA is below stock for every distil** — 79.0,
76.9, 77.4, 76.6 against 80.7 — and four compositions failed to recover it.
HumanEval tracks nothing we controlled.

### 5.2 Four hypotheses about TruthfulQA, all wrong

1. *TriviaQA taught confident recall, the opposite of what TruthfulQA rewards.*
   Replaced with 876 gold-labelled misconception items. The metric went **down**.
2. *SFT erodes calibration in proportion to training volume.* The damage turned
   out to be confined to abstention; general accuracy was untouched.
3. *The v1→v2 drop was itself a hedging effect.* v2 is actually **better** on
   hedge questions than v1.
4. *Rejection sampling suppresses abstention, so add abstention data.* Best
   evidence of the four, and 527 abstention samples still moved TruthfulQA down
   1.6 points.

**The diagnosis that survived scrutiny:** v1 matches the stock model exactly on
ordinary questions (81.3% vs 81.4%) and **loses 11 points on questions whose
correct answer is "I don't know"** (65.0% vs 76.1%) — 19 lost against 6 gained,
p ≈ 0.016. Every training target is a confident correct answer, because that is
what rejection sampling keeps; the student never sees appropriate uncertainty.
The mechanism is well evidenced. The obvious fix still didn't work, and neither
did three other mixtures. Fixing it means changing the **filter**, not the mix.

**Removing the length bias didn't help either, and it should have.** Every
model up to v4 trained on whatever fit under a 1024-token cap, which was not a
random filter: v1 trained on **1,758 traces of median 678 tokens and never saw
1,735 of median 1,449**. The chunkwise path made 2048-token training fit, so v4
lifted the cap with sample count, step count, slices, mixture and every
hyperparameter held constant, and a `max_samples` cap added so that removing
the length limit could not smuggle in extra volume. Median trace length went
686 → 1,010, p90 963 → 1,758. **The prediction was on the record before
measuring: HumanEval improves most**, because §29 had independently shown that
shortening code reasoning costs 5.4 points. HumanEval **fell 1.2 points**. All
three deltas lean negative and all sit inside their noise bands. This was the
best-motivated data experiment of the four — a bias that was measured rather
than guessed — and it changed nothing. **Data composition is closed.**

### 5.3 v5: logit distillation made the model unable to stop

v5 was the only remaining lever that differed in kind rather than degree. Every
earlier run minimised cross-entropy against **one sampled completion** per
prompt: the student saw which token the teacher emitted and nothing about how
confident it was or what it nearly said. v5 trains against the distribution:

    L = 0.3 · CE(student, sampled token) + 0.7 · KL(teacher ‖ student)

over the teacher's top-64, renormalised. No regeneration was needed — the
traces already existed, so teacher-forcing them yields every distribution in
one forward pass at 1.3 s per 1,000-token sample against the ~6 s it cost to
sample that trace originally. Everything else was held at v1's settings: same
1,639 samples, same slices, 3,200 steps, same rank, layers and learning rate.

**Result: 77.2% MMLU — the first distil to score below the stock model — in
69,897 seconds against v1's 13,007 on the same 1,000 questions.**

**The 5.4x wall clock is the finding, not the accuracy.** At identical size,
bit width and decode settings, v5 did not become confused; it became unable to
stop. Under a 2,048-token budget that converts directly into wrong answers
through truncation — the same termination axis as §4, driven the wrong way by
the objective. The remaining two benchmarks were abandoned rather than spend
multiple days measuring a model already known to have failed.

The mechanism is what the objective literally asks for. CE trains the student
toward the single token the teacher committed to. KL trains it to reproduce the
teacher's *entire* distribution at every step — including the teacher's
residual uncertainty about **whether to stop reasoning**. A 35B teacher can
carry that uncertainty and still terminate; a 9B student at 4.7 bits cannot.

**This is a result, not a bug.** A worse model from a silent misalignment looks
identical to a worse model from a bad objective, so three failure modes were
checked before accepting it:

| check | outcome |
|---|---|
| Vocabulary identity — the student's tokenizer feeds *both* teachers, and Qwen3.6's `tokenizer.json` **does** differ (different hash, +18 bytes) | vocabularies **index-identical** (248,044 entries; `vocab` and `merges` equal). Difference is the pre-tokenizer regex (`\p{M}` in the letter class), which changes how text encodes, never what an id means |
| Top-64 mass coverage — asserted at 99.9% from the literature, never measured on these shards | measured over all 898,322 stored positions: **99.93%** knowledge, **99.96%** code. Renormalising moves mean top-1 from 0.8849 to 0.8853 |
| Span alignment and coverage | **1,639/1,639** train and **33/33** valid rows present, **zero** uncovered target positions |

The objective was implemented correctly and faithfully, and it made the model
worse.

**One latent bug surfaced during those checks and is now fixed.**
`make_batches` filled rows lacking teacher logits with `ids = 0`,
`lps = -30.0`, and left those positions **inside the loss mask**. A uniform
`-30` renormalises to a uniform distribution over 64 copies of token id 0 — i.e.
certainty on token 0 — so such rows trained toward garbage rather than "CE
alone" as the adjacent print statement claimed. Complete coverage meant it never
fired on this run, but it would silently poison any run with partial
extraction. The KL term now carries an explicit coverage mask; verified that an
uncovered row contributes exactly `0.3 × CE`.

### 5.4 v6: the teacher does not have the behaviour we wanted to distill

§5.2 localised the TruthfulQA regression precisely — it lives entirely in
questions whose correct answer is "I don't know" — and blamed the **filter**:
rejection sampling keeps only confident correct answers, so the student never
sees warranted uncertainty. v3 tried to fix that by *adding* 527 SQuAD-v2
unanswerable items and lost ground; that is reading comprehension ("the passage
does not say"), not epistemic hedging.

v6 changed the filter instead. When the knowledge teacher answers an open-ended
question **wrong**, that is by construction a question where confident
assertion was not warranted — and v1 discarded it. Those 122 prompts went back
to the teacher with its confidence made explicit, keeping the trace only if it
then declined, paired with the *original* prompt. MCQ failures were excluded on
purpose: MMLU does not penalise guessing, so teaching abstention there would
risk the one metric v1 wins.

**v6 was never trained.** The slice could not support the experiment:

| | count |
|---|---:|
| rejected open-ended traces, re-asked | 121 |
| **teacher answered confidently anyway** | **99 (82%)** |
| declined — usable hedge samples | 22 |
| of those, surviving v1's 1,024-token cap | **0** |

§5's threshold was set in advance at ~60 samples. It came in at 22, and at
**0** after the length cap, because the 22 run 1,295–3,158 tokens (median
1,900) — the teacher deliberates at length before admitting ignorance, the same
effect that made "be brief" produce *longer* abstention traces.

**The number that matters is 99, not 22.** On questions it had already answered
wrong, invited explicitly to say it did not know, Qwen3.6-35B re-asserted a
confident answer **82% of the time**. The re-ask did not instruct it to decline
— that would have made the verifier a rubber stamp — it asked for an honest
confidence judgement and left both outcomes open.

This closes the abstention line for a better reason than v3's failure did.
**Sequence-level distillation can only transmit what the teacher emits, and
this teacher does not emit calibrated uncertainty.** §5.2's diagnosis was right
about the mechanism and wrong to assume the remedy was available: the discarded
pool is not full of hedges waiting to be recovered, it is full of confident
errors.

Cost: about an hour of teacher generation, no training. A stopping rule
satisfied on evidence is cheaper than a sixth null result. The 22 verified
traces are kept at `runs/v6/teacher/hedge.jsonl`; the supply failed, not the
method.

*One bug it surfaced:* the first dry run kept 0 of 16 and called them all
"answered confidently". They were **truncated** — at a 1,024-token budget the
teacher spent the whole budget inside `<think>` and returned an empty answer,
and an empty answer is not a confident one. §4's truncation finding,
reappearing inside our own pipeline. Now uses the teacher's own 2,048 budget
with escalating retry, and counts no-answer separately.

### 5.5 Other things that failed

**Telling the code teacher to be brief cost 5.4 points of HumanEval.** It
worked as instructed — median reasoning fell 1,328 → 1,034 characters and the
code slice grew 421 → 530 samples. More data of a worse kind. (Asked for
shorter *abstention* traces, meanwhile, the model produced longer ones, 1,318 →
1,524 tokens. Brevity instructions don't reliably work here.)

**QLoRA is worse than bf16 LoRA here.** Planned on the theory that a 4.9 GB
base would free memory for longer sequences. Measured: **49.0 GB peak against
bf16's 47.3, and 26 s/iter against 10.** Dequantizing every pass costs more than
smaller weights save, and the activation memory that dominates is bf16 either
way.

**Ollama/GGUF export doesn't work for this architecture.** Four real
incompatibilities were fixed — architecture naming, transformers-v5 tensor
prefixes, MLX storing conv1d as `(out, kernel, in)` where PyTorch expects
`(out, in, kernel)`, and a config declaring an MTP head the checkpoint doesn't
contain — after which it loads and emits gibberish. An MLX→GGUF problem, not a
distillation one.

---

## 6. The engineering result

Training memory was pinned at 1024-token sequences, which rejected **46% of
every trace generated — six times what wrong answers cost.** The cause was not
the hardware: `mlx_lm.models.gated_delta` trains these layers through a
**sequential Python loop over every timestep**, because the fast Metal kernel
used at inference has no VJP. Its own docstring calls it a "reference
implementation for prompt prefill". Measured: ~8 MB of autograd state per token
per layer.

The gated delta rule has a chunkwise parallel form
([arXiv:2406.06484](https://arxiv.org/abs/2406.06484),
[arXiv:2412.06464](https://arxiv.org/abs/2412.06464)) that keeps O(T/C) states
instead of O(T). One GDN layer, forward + backward, chunkwise at C=64:

| seq | mlx-lm sequential | chunkwise | gain |
|---:|---:|---:|---|
| 512 | 4.29 GB / 0.36 s | 0.33 GB / 0.04 s | 13x mem, 10x speed |
| 1024 | 10.18 GB / 1.18 s | 0.99 GB / 0.09 s | 10x mem, 13x speed |
| 2048 | 31.44 GB / 4.21 s | 1.74 GB / 0.22 s | **18x mem, 19x speed** |

Fusing the per-chunk triangular solve into a Metal kernel with a hand-written
VJP adds another 2.1–2.7x. Cumulative against stock mlx-lm at 4096 tokens:
**108.33 GB / 39.02 s → 4.60 GB / 0.29 s — 24x memory, 135x speed.**

Outputs match the sequential reference to ~3e-7 relative in fp32 across all five
gradients, and the inference kernel at 100% top-1 on the full 9B.

Two details that mattered. `mx.linalg.tri_inv` is CPU-only with no VJP, so it
cannot appear in a training graph — the unit-triangular inverse is built from
matmuls by 2x2 block recursion in log2(C) levels. And the textbook derivation
divides by the cumulative decay, which underflows on strongly decaying heads and
sends the intermediate writes to infinity; **carrying bounded ratios instead is
what makes it work on a real model.**

For the pipeline: the same traces already on disk now yield **4,876 training
samples instead of 2,704**, length rejections fall from 2,391 to 219, peak
memory at 2048 is 41.8 GB (below the 47.3 GB that 1024 used to require), and
end-to-end throughput doubled — **55 → 111 tokens/s**.

### Upstream

**[mlx-lm#1870](https://github.com/ml-explore/mlx-lm/pull/1870) — chunkwise
gated delta — is open and reported MERGEABLE.** `gated_delta_ops` becomes a
dispatcher: scalar gating without a mask takes the chunkwise path, everything
else goes to the existing loop, renamed `gated_delta_sequential` to say what it
is. Five model families train through that code — `qwen3_5`, `qwen3_next`,
`kimi_k3`, `kimi_linear`, `bailing_moe_v3`.

**A second branch, `gated-delta-fused-solve`, is pushed and waiting on it.**
Profiling showed the per-chunk inverse was 5.07 ms of ~9.3 ms and is used
exactly once as `inv @ rhs` — so solve rather than invert. Stacking a PR on an
unreviewed PR is poor practice, so it waits.

**GPU `tri_inv` in mlx core was deliberately not attempted.**
[issue #1392](https://github.com/ml-explore/mlx/issues/1392) is open and
[PR #4375](https://github.com/ml-explore/mlx/pull/4375) already tried it and was
closed as a draft. Its numbers kill the obvious approach — an MPS-backed Metal
inverse is **20x slower than CPU at 64x64** — and it is blocked on an unmade
maintainer decision, not on someone writing code.

---

## 7. Process lessons

**Get the per-item control before theorising.** Three wrong hypotheses about
TruthfulQA died within minutes of the per-question files arriving. Roughly 15
hours of compute went into v2, aimed at a cause the control would have ruled
out in advance.

**Check the harness before believing the numbers.** TruthfulQA lists the
correct answer first in all 817 items; presenting them in dataset order made
every gold answer "A", so the local metric was measuring position bias — of
v1's 177 correct answers, 177 were "A" picks. Fixed by shuffling under a
per-item seed; the oMLX harness already shuffled, so its numbers stood.

**Compare like with like.** An early read called v1 a failure on code by
comparing a 4.72-bit model against a bf16 one. Against the model it actually
replaces it was +12.2 points.

**Change one variable.** v2 kept 3,200 steps while the dataset grew 37%, so it
was simultaneously better-fed and less-trained and nothing can be concluded from
it. v4 and v5 were both built the other way, which is why their null results are
worth something.

**Write the prediction down before measuring.** v4's prediction was specific
(HumanEval improves most), falsifiable, and wrong. A vaguer one would have
survived the result and taught nothing.

**Validation loss is not a proxy for benchmark quality.** v2 had the best train
and val loss of any run (0.116 / 0.267) and the worst benchmarks of the first
four. v5 posted the best val loss on its own objective (0.180) and scored below
the stock model.

**Rule out the bug before believing the result — and before dismissing it.**
v5's three post-hoc checks each took minutes and each could have turned a
publishable negative result into an embarrassing one. One of them (Qwen's
differing tokenizer hash) looked alarming and was benign; a fourth turned up a
real latent bug that had not fired.

**Wall clock is a measurement.** v5's accuracy said "worse model". Its 5.4x
runtime said *why*. The termination finding in §4 came from the same habit of
reading the timing column.

**Serialize GPU work.** Running conversions alongside training killed both, and
then cost a second training start via memory pressure. The rule was written
after the first time and broken anyway.

**Ask why a constraint exists.** Six sections of the decision log are careful
work inside a 1024-token limit — trimming data, choosing what to sacrifice. The
limit was a sequential loop in a dependency, and removing it took a day. It is
the highest-leverage thing in the project, and it was not on the plan.

---

## 8. Conclusions, and what is actually left

**Three axes are closed, and one was never opened.** Data composition failed
four times (§5.1), including once against a bias that was measured rather than
guessed. The training objective failed once, worse than anything else (§5.3).
The filter — the one remaining idea with a clean mechanism behind it — turned
out to have no supply to draw on, because the teacher does not express
calibrated uncertainty (§5.4).

**v1 is the result**, and it is a good one: a one-time repair of what
quantization broke, +5.5 MMLU and +3.0 HumanEval at the same size and bit
width, plus a 15–33% speedup from the same mechanism.

**What was never varied is the adapter.** Every run — v1 through v6 — used
rank 32 over the top 16 of 32 layers, chosen once before v1 and never revisited
while five other things were swept. §5.3 concluded the residual gap is
"capacity, not recipe", but *adapter* capacity is the one capacity knob the
project never tested. That asymmetry is the most obvious thing left.

In descending order of expected value:

1. **Vary the adapter, not the data.** Rank 32 → 64 and top-16 → all 32 layers,
   holding data, steps and learning rate fixed. Cheap (same pipeline, one
   config change), directly tests the conclusion five runs have been leaning
   on, and is the only untouched axis. If capacity is genuinely the limit, this
   is where it shows; if it changes nothing, "capacity, not recipe" is
   confirmed properly rather than by elimination.
2. **Train the domains in sequence rather than mixed.** Also never tried. The
   40/25/35 mixture assumes the three slices do not interfere, and nothing
   tested that assumption — TruthfulQA falling below stock in *every* run while
   MMLU and HumanEval rise is at least consistent with interference.
3. **Land mlx-lm#1870, then open the fused-solve PR.** The engineering work is
   done and tested; only the review queue stands between it and five model
   families training an order of magnitude cheaper.
4. **If abstention is revisited, change the teacher, not the filter.** §5.4
   shows the supply problem is in Qwen3.6-35B itself. A teacher that actually
   hedges — or a non-distillation method such as calibration tuning on the
   student's own logits — is the only route that could work.
5. **Don't try another data mixture, and don't retry logit KD unchanged.** Four
   and one failures respectively. If logit KD is retried, fix the termination
   signal first (lower KL weight, or exclude the KL term near end-of-reasoning
   tokens) and report truncation rate and wall clock, not just accuracy.
6. **Cut Cross-Entropy** ([arXiv:2411.09009](https://arxiv.org/abs/2411.09009)).
   A 248,044-token vocabulary costs ~3.8 GB in logits at 2048 — the largest
   remaining implementation-level cost.
