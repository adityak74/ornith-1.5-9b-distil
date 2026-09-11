# Ornith-1.5-9B distillation: what worked, what didn't

A record of the whole effort — four distilled models, one shipped, one upstream
PR open with a second branch behind it, and a pile of wrong hypotheses. Written
for whoever picks this up next, including future me.

Detail for every decision is in [`DECISIONS.md`](DECISIONS.md) (36 sections);
this is the summary.

---

## The result

| model | size | MMLU | TruthfulQA | HumanEval |
|---|---:|---:|---:|---:|
| Ornith-1.5-9B-oQ4 (stock) | 4.9 GB | 78.0% | **80.7%** | 87.8% |
| Ornith-1.5-9B-oQ8 | 8.9 GB | 83.1% | 80.7% | 88.4% |
| Ornith-1.5-9B bf16 | 17 GB | 82.6% | 80.5% | **91.5%** |
| **distil-oQ4 (shipped)** | **4.9 GB** | **83.5%** | 79.0% | 90.8% |
| distil-v2-oQ4 | 4.9 GB | 83.1% | 76.9% | 85.4% |
| distil-v3-oQ4 | 4.9 GB | 83.6% | 77.4% | 87.2% |
| distil-v4-oQ4 | 4.9 GB | 83.2% | 76.6% | 89.6% |
| Ornith-1.5-35B-A3B-4bit (teacher) | 18 GB | 83.0% | 86.4% | 93.3% |
| Qwen3.6-35B-A3B-4bit (teacher) | 19 GB | **89.3%** | **89.2%** | 93.3% |

All measured on the oMLX server, same protocol for every row.

**v1 shipped**: +5.5 MMLU and +3.0 HumanEval over the 4-bit model it replaces,
at identical size and bit width. It beats the 1.8x larger oQ8 on both, edges
out the 3.5x larger bf16 base on MMLU, and matches its own 35B teacher on MMLU.
It is also 33% faster on HumanEval and 15% on MMLU.

**v2, v3 and v4 were not shipped.** None of them beat v1. That is most of this
document.

---

## What worked

**Two teachers, split by measurement.** Qwen3.6-35B is 6.3 points better on
MMLU; Ornith-35B ties it on HumanEval while sampling 4.6x faster. Routing
knowledge to one and code to the other was cheaper than using the stronger
teacher for everything, and the MMLU gain came through.

**Rejection sampling.** Keeping only verifiably correct traces — gold letter
for multiple choice, gold aliases for open questions, and generated code
actually executed against the source dataset's tests. Qwen is 89.3% on MMLU, so
roughly one in nine unfiltered knowledge traces would have been a confidently
argued wrong answer. (It also costs TruthfulQA — see below. Both are true.)

**Quantizing after training, not before.** Training LoRA on the bf16 base and
quantizing the fused result meant the quantizer started from an improved
checkpoint instead of compounding two lossy steps.

**Reproducing the quantization map exactly.** oQ4 is 4-bit affine at group size
64 with 120 modules promoted to 5/6/8 bits, and the whole map is recoverable
from the checkpoint's `config.json`. Replaying it made the comparison
like-for-like instead of "our q4 vs their oQ4".

**The chunkwise gated-delta training path.** The biggest engineering result of
the project, and it came from asking why a constraint existed instead of
designing around it. See below.

---

## What didn't

**Four data experiments, four failures.** v1 is the baseline and nothing beat
it:

| run | what it changed | MMLU | TruthfulQA | HumanEval |
|---|---|---:|---:|---:|
| v1 | — | **83.5%** | 79.0% | **90.8%** |
| v2 | misconception slice replaces TriviaQA, brief code | 83.1% | 76.9% | 85.4% |
| v3 | abstention slice + v1's code traces restored | 83.6% | 77.4% | 87.2% |
| v4 | length bias removed, everything else held | 83.2% | 76.6% | 89.6% |

Volume was not the answer either: v1 used 1,672 samples, v2 2,286, v3 4,075,
and both larger runs lost ground.

**MMLU is saturated at ~83.5** — 83.5, 83.1, 83.6, 83.2 across wildly different
mixtures, volumes and length distributions, and v1 already matches its own 35B
teacher there (83.0). **TruthfulQA is below stock for every distil** — 79.0,
76.9, 77.4, 76.6 against 80.7 — and four compositions failed to recover it.
HumanEval tracks nothing we controlled.

**Four hypotheses about TruthfulQA, all wrong.**

1. *TriviaQA taught confident recall, the opposite of what TruthfulQA rewards.*
   Replaced it with 876 gold-labelled misconception items. The metric went
   **down**.
2. *SFT erodes calibration in proportion to training volume.* The damage turned
   out to be confined to abstention; general accuracy was untouched.
3. *The v1→v2 drop was itself a hedging effect.* v2 is actually **better** on
   hedge questions than v1.
4. *Rejection sampling suppresses abstention, so add abstention data.* This one
   had the best evidence behind it — see below — and 527 abstention samples
   still moved TruthfulQA down 1.6 points.

The diagnosis that survived scrutiny: **v1 matches the stock model exactly on
ordinary questions (81.3% vs 81.4%) and loses 11 points on questions whose
correct answer is "I don't know"** (65.0% vs 76.1%), 19 lost against 6 gained,
p ≈ 0.016. Every training target is a confident correct answer, because that is
what rejection sampling keeps; the student never sees appropriate uncertainty.
The mechanism is well evidenced. The obvious fix still didn't work, and neither
did three other mixtures.

**Removing the length bias didn't help either, and it should have.** Every
model up to v4 trained on whatever fit under a 1024-token cap, which was not a
random filter: v1 trained on **1,758 traces of median 678 tokens and never saw
1,735 of median 1,449**. The chunkwise path made 2048-token training fit, so v4
lifted the cap with sample count, step count, slices, mixture and every
hyperparameter held constant — a `max_samples` cap was added so that removing
the length limit couldn't smuggle in extra volume. Median trace length went
686 → 1,010, p90 963 → 1,758. The prediction was on the record before
measuring: **HumanEval improves most**, because §29 had independently shown
that shortening code reasoning costs 5.4 points. HumanEval **fell 1.2 points**.
All three deltas lean negative and all sit inside their noise bands. This was
the best-motivated data experiment of the four — a bias that was measured
rather than guessed — and it changed nothing. **Data composition is closed as
an axis.**

**Telling the code teacher to be brief cost 5.4 points of HumanEval.** It
worked as instructed: median reasoning fell from 1,328 to 1,034 characters and
the code slice grew from 421 to 530 samples. More data of a worse kind. Code is
where reasoning pays — which is exactly why v4's failure is surprising. (Asked
for shorter *abstention* traces, meanwhile, the model produced longer ones —
1,318 to 1,524 tokens. Brevity instructions don't reliably work here.)

**QLoRA is worse than bf16 LoRA here.** Planned on the theory that a 4.9 GB
base would free memory for longer sequences. Measured: **49.0 GB peak against
bf16's 47.3, and 26 s/iter against 10.** Dequantizing every pass costs more
than smaller weights save, and the activation memory that dominates is bf16
either way. Dropped after measuring.

**Ollama/GGUF export doesn't work for this architecture.** Four real
incompatibilities were fixed — architecture naming, transformers-v5 tensor
prefixes, MLX storing conv1d as `(out, kernel, in)` where PyTorch expects
`(out, in, kernel)`, and a config declaring an MTP head the checkpoint doesn't
contain — after which it loads and emits gibberish. It's an MLX→GGUF problem,
not a distillation one.

---

## The engineering result

Training memory was pinned at 1024-token sequences, which rejected **46% of
every trace generated — six times what wrong answers cost.** The cause was not
the hardware: `mlx_lm.models.gated_delta` trains these layers through a
**sequential Python loop over every timestep**, because the fast Metal kernel
used at inference has no VJP. Its own docstring calls it a "reference
implementation for prompt prefill". Measured: ~8 MB of autograd state per token
per layer.

The gated delta rule has a chunkwise parallel form ([arXiv:2406.06484](https://arxiv.org/abs/2406.06484),
[arXiv:2412.06464](https://arxiv.org/abs/2412.06464)) that keeps O(T/C) states
instead of O(T). One GDN layer, forward + backward, chunkwise at C=64:

| seq | mlx-lm sequential | chunkwise | gain |
|---:|---:|---:|---|
| 512 | 4.29 GB / 0.36 s | 0.33 GB / 0.04 s | 13x mem, 10x speed |
| 1024 | 10.18 GB / 1.18 s | 0.99 GB / 0.09 s | 10x mem, 13x speed |
| 2048 | 31.44 GB / 4.21 s | 1.74 GB / 0.22 s | **18x mem, 19x speed** |

Then fusing the per-chunk triangular solve into a Metal kernel with a
hand-written VJP adds another 2.1–2.7x. Cumulative against stock mlx-lm at
4096 tokens: **108.33 GB / 39.02 s → 4.60 GB / 0.29 s — 24x memory, 135x
speed.**

Outputs match the sequential reference to ~3e-7 relative in fp32 across all
five gradients, and the inference kernel at 100% top-1 on the full 9B.

Two details that mattered. `mx.linalg.tri_inv` is CPU-only with no VJP, so it
cannot appear in a training graph — the unit-triangular inverse is built from
matmuls by 2x2 block recursion in log2(C) levels. And the textbook derivation
divides by the cumulative decay, which underflows on strongly decaying heads
and sends the intermediate writes to infinity; carrying bounded ratios instead
is what makes it work on a real model.

For the pipeline: the same traces already on disk now yield **4,876 training
samples instead of 2,704**, length rejections fall from 2,391 to 219, peak
memory at 2048 is 41.8 GB (below the 47.3 GB that 1024 used to require), and
end-to-end training throughput doubled — **55 → 111 tokens/s**, so v4 trained
2.79M tokens in 7.1 hours against v1's 1.76M in 8.6.

### Upstream

**[mlx-lm#1870](https://github.com/ml-explore/mlx-lm/pull/1870) — chunkwise
gated delta — is open and reported MERGEABLE.** `gated_delta_ops` becomes a
dispatcher: scalar gating without a mask takes the chunkwise path, everything
else goes to the existing loop, renamed `gated_delta_sequential` to say what it
is. Five model families train through that code — `qwen3_5`, `qwen3_next`,
`kimi_k3`, `kimi_linear`, `bailing_moe_v3`.

**A second branch, `gated-delta-fused-solve`, is pushed and waiting on it.**
Profiling showed the per-chunk inverse was 5.07 ms of ~9.3 ms and is used
exactly once as `inv @ rhs` — so solve rather than invert. Substitution in one
Metal kernel, with `mx.custom_function` supplying the gradient autodiff cannot
derive through a hand-written kernel. Stacking a PR on an unreviewed PR is poor
practice, so it waits.

**GPU `tri_inv` in mlx core was deliberately not attempted.** Scoped as a third
contribution, then dropped once the ground truth turned up:
[issue #1392](https://github.com/ml-explore/mlx/issues/1392) is open and
[PR #4375](https://github.com/ml-explore/mlx/pull/4375) already tried it and was
closed as a draft. Its numbers kill the obvious approach — an MPS-backed Metal
inverse is **20x slower than CPU at 64x64** and only wins past 4096x4096 — and
it is blocked on an unmade maintainer decision, not on someone writing code.
Our own kernel is counter-evidence for the narrow case, but a triangular solve
is a far easier problem than general LU inversion. Worth a comment on the
issue; not worth a duplicate PR.

---

## The finding worth publishing

**Quantization damages a reasoning model's ability to stop more than its
ability to reason.**

On a strict harness the stock 4-bit model **never reaches an answer on 24% of
HumanEval problems**. The distilled model cuts that to 12%, and **22 of the 28
problems it gains are ones where the stock model ran out of budget
mid-reasoning**. Mean generation drops from 1,721 to 1,414 tokens. It is not a
scoring artefact: it reproduces as a **33% wall-clock reduction on a harness
that doesn't penalise truncation at all**. And the same weights score 82.3% or
88.4% on HumanEval depending only on whether the budget is 2048 or 4096 — any
benchmark of a reasoning model that doesn't report its truncation rate is
partly reporting its own token budget.

Distillation on traces that terminate cleanly repairs most of it. That is what
v1 is.

---

## Process lessons

**Get the per-item control before theorising.** Three wrong hypotheses about
TruthfulQA died within minutes of the per-question files arriving. Roughly 15
hours of compute went into v2, aimed at a cause the control would have ruled
out in advance.

**Check the harness before believing the numbers.** TruthfulQA lists the
correct answer first in all 817 items; presenting them in dataset order made
every gold answer "A", so the local metric was measuring position bias — of v1's
177 correct answers, 177 were "A" picks. Fixed by shuffling under a per-item
seed; the oMLX harness already shuffled, so its numbers stood.

**Compare like with like.** An early read called v1 a failure on code by
comparing a 4.72-bit model against a bf16 one. Against the model it actually
replaces it was +12.2 points.

**Change one variable.** v2 kept 3,200 steps while the dataset grew 37%, so it
was simultaneously better-fed and less-trained and nothing can be concluded
from it. v4 was built the other way — one variable moved, a `max_samples` cap
added so volume couldn't leak in — which is why its null result is worth
something.

**Write the prediction down before measuring.** v4's prediction was specific
(HumanEval improves most), falsifiable, and wrong. A vaguer one would have
survived the result and taught nothing.

**Validation loss is not a proxy for benchmark quality.** v2 had the best train
and val loss of any run (0.116 / 0.267) and the worst benchmarks. v4's val loss
was 0.305 and it beat v2 on every metric.

**Serialize GPU work.** Running conversions alongside training killed both, and
then cost a second training start via memory pressure. The rule was written
after the first time and broken anyway.

**Ask why a constraint exists.** Six sections of this log are careful work
inside a 1024-token limit — trimming data, choosing what to sacrifice. The
limit was a sequential loop in a dependency, and removing it took a day. It is
the highest-leverage thing in the project, and it was not on the plan.

---

## If someone continues

1. **Offline top-k logit distillation.** The only remaining lever, and the only
   untried change that differs in kind rather than degree. Every run so far
   trained on **one sampled completion per prompt**; logit KD trains against the
   teacher's full next-token distribution, which the shared 248,044-token
   vocabulary — verified byte-identical between the two Ornith models — makes
   possible. It needs a custom trainer, roughly doubles generation cost, and is
   off-policy, which the literature favours for knowledge benchmarks. Work on
   this is starting now.
2. **Cut Cross-Entropy.** A 248,044-token vocabulary costs ~3.8 GB in logits at
   2048; [arXiv:2411.09009](https://arxiv.org/abs/2411.09009) removes it. The
   largest remaining implementation-level cost, and it compounds with (1),
   which is logit-heavy by construction.
3. **Don't try another data mixture.** Four attempts, four failures, including
   one aimed at a bias that was measured rather than guessed. MMLU is saturated
   and TruthfulQA doesn't respond to composition — its abstention deficit is
   caused by rejection sampling itself, not by any particular slice, so fixing
   it means changing the filter or the training signal.
4. **Land mlx-lm#1870, then open the fused-solve PR.** The engineering work is
   done and tested; only the review queue is between it and five model families
   training an order of magnitude cheaper.
