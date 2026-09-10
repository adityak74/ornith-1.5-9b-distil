# Ornith-1.5-9B distillation: what worked, what didn't

A record of the whole effort — three distilled models, one shipped, two upstream
contributions, and four wrong hypotheses. Written for whoever picks this up
next, including future me.

Detail for every decision is in [`DECISIONS.md`](DECISIONS.md) (34 sections);
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
| Ornith-1.5-35B-A3B-4bit (teacher) | 18 GB | 83.0% | 86.4% | 93.3% |
| Qwen3.6-35B-A3B-4bit (teacher) | 19 GB | **89.3%** | **89.2%** | 93.3% |

All measured on the oMLX server, same protocol for every row.

**v1 shipped**: +5.5 MMLU and +3.0 HumanEval over the 4-bit model it replaces,
at identical size and bit width. It beats the 1.8x larger oQ8 on both, edges
out the 3.5x larger bf16 base on MMLU, and matches its own 35B teacher on MMLU.
It is also 33% faster on HumanEval and 15% on MMLU.

**v2 and v3 were not shipped.** Both were worse. That is most of this document.

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
argued wrong answer.

**Quantizing after training, not before.** Training LoRA on the bf16 base and
quantizing the fused result meant the quantizer started from an improved
checkpoint instead of compounding two lossy steps.

**Reproducing the quantization map exactly.** oQ4 turned out to be 4-bit affine
at group size 64 with 120 modules promoted to 5/6/8 bits, and the whole map is
recoverable from the checkpoint's `config.json`. Replaying it on the distilled
model made the comparison like-for-like instead of "our q4 vs their oQ4".

**The chunkwise gated-delta training path.** The biggest engineering result,
and it came from asking why a constraint existed instead of designing around
it. See below.

---

## What didn't

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
The mechanism is well evidenced. The obvious fix still didn't work.

**More data never helped.** v1 used 1,672 samples, v2 2,286, v3 4,075 — MMLU
saturated at v1's level (83.5 → 83.1 → 83.6) and never moved again, while both
larger runs lost ground elsewhere.

**Telling the code teacher to be brief cost 5.4 points of HumanEval.** It
worked as instructed: median reasoning fell from 1,328 to 1,034 characters and
the code slice grew from 421 to 530 samples. More data of a worse kind. Code is
where reasoning pays.

**Brevity instructions don't work on this model anyway.** Asked for shorter
abstention traces, it produced *longer* ones — 1,318 to 1,524 tokens. The
knowledge teacher ignored a 150-word limit too.

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
the hardware.

`mlx_lm.models.gated_delta` trains these layers through a **sequential Python
loop over every timestep**, because the fast Metal kernel used at inference has
no VJP. Its own docstring calls it a "reference implementation for prompt
prefill". Measured: ~8 MB of autograd state per token per layer.

The gated delta rule has a chunkwise parallel form ([arXiv:2406.06484](https://arxiv.org/abs/2406.06484),
[arXiv:2412.06464](https://arxiv.org/abs/2412.06464)) that keeps O(T/C) states
instead of O(T). Implementing it, then fusing the per-chunk triangular solve
into a Metal kernel with a hand-written VJP:

| seq | mlx-lm today | after both | gain |
|---:|---:|---:|---|
| 1024 | 10.18 GB / 1.16 s | 1.00 GB / 0.03 s | 10x mem, 39x speed |
| 2048 | 31.44 GB / 4.20 s | 1.75 GB / 0.09 s | 18x mem, 47x speed |
| 4096 | 108.33 GB / 39.02 s | 4.60 GB / 0.29 s | **24x mem, 135x speed** |

Outputs match the sequential reference to ~5e-7 and the inference kernel at
100% top-1 on the full 9B.

Two details that mattered. `mx.linalg.tri_inv` is CPU-only with no VJP, so it
cannot appear in a training graph — the inverse is built from matmuls by block
recursion, which also parallelizes over the batch. And the textbook derivation
divides by the cumulative decay, which overflows fp32 on strongly decaying
heads; carrying bounded ratios instead is what makes it work on a real model.

For the pipeline: the same traces already on disk now yield **4,876 training
samples instead of 2,704**, with length rejections falling from 2,391 to 219.

Upstream: [mlx-lm#1870](https://github.com/ml-explore/mlx-lm/pull/1870) is open
and mergeable; the fused-solve branch waits for it. Five model families train
through that code — `qwen3_5`, `qwen3_next`, `kimi_k3`, `kimi_linear`,
`bailing_moe_v3`.

---

## The finding worth publishing

**Quantization damages a reasoning model's ability to stop more than its
ability to reason.**

On a strict harness the stock 4-bit model **never reaches an answer on 24% of
HumanEval problems**. The distilled model cuts that to 12%, and **22 of the 28
problems it gains are ones where the stock model ran out of budget
mid-reasoning**. Mean generation drops from 1,721 to 1,414 tokens.

It is not a scoring artefact: it reproduces as a **33% wall-clock reduction on
a harness that doesn't penalise truncation at all**. And the same weights score
82.3% or 88.4% on HumanEval depending only on whether the budget is 2048 or
4096 — any benchmark of a reasoning model that doesn't report its truncation
rate is partly reporting its own token budget.

Distillation on traces that terminate cleanly repairs most of it. That is what
v1 is.

---

## Process lessons

**Get the per-item control before theorising.** Three wrong hypotheses about
TruthfulQA died within minutes of the per-question files arriving. Roughly 15
hours of compute went into v2, aimed at a cause the control would have ruled
out immediately.

**Check the harness before believing the numbers.** TruthfulQA lists the
correct answer first in all 817 items; presenting them in dataset order made
every gold answer "A", so the local metric was measuring position bias — of v1's
177 correct answers, 177 were "A" picks. Fixed by shuffling under a per-item
seed. The oMLX harness already shuffled, so its numbers stood.

**Compare like with like.** An early read called v1 a failure on code by
comparing a 4.72-bit model against a bf16 one. Against the model it actually
replaces it was +12.2 points.

**Hold epochs constant, not iterations.** v2 kept 3,200 steps while the dataset
grew 37%, so it was simultaneously better-fed and less-trained, and the two
cannot be separated from that run.

**Validation loss is not a proxy for benchmark quality.** v2 had the best train
and val loss of any run (0.116 / 0.267) and the worst benchmarks.

**Serialize GPU work.** Running conversions alongside training killed both, and
then cost a second training start via memory pressure. The rule was written
after the first time and broken anyway.

**Ask why a constraint exists.** Six sections of this log are careful work
inside a 1024-token limit — trimming data, choosing what to sacrifice. The
limit was a sequential loop in a dependency, and removing it took a day.

---

## If someone continues

1. **Run v4 at 2048.** Every model so far trained on the *shortest 48%* of what
   the teachers produced. That was never a deliberate choice, and it is now
   removed. Whether more data helps is genuinely open — but this is different
   data, not just more.
2. **Cut Cross-Entropy.** A 248,044-token vocabulary costs ~3.8 GB in logits at
   2048; [arXiv:2411.09009](https://arxiv.org/abs/2411.09009) removes it. Now
   the largest remaining implementation-level cost.
3. **Offline top-k logit distillation.** All three teachers share an identical
   248,044-token vocabulary — verified, byte-identical between the two Ornith
   models. Every run so far trained on one sampled completion per prompt;
   logit-level KD trains against the teacher's full distribution. It is the
   only remaining change that is different in kind rather than degree.
4. **Don't chase TruthfulQA with data.** Four attempts, four failures, one of
   them well evidenced. Something other than the training mixture is going on.
