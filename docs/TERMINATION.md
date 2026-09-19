# Quantization breaks stopping before it breaks reasoning

**A 9B reasoning model quantized to 4 bits does not get much worse at
reasoning. It gets worse at deciding it is finished — and on a benchmark with a
token budget, those look identical.**

Measured while distilling Ornith-1.5-9B on Apple silicon. Full decision log in
[`DECISIONS.md`](../DECISIONS.md); the project write-up is
[`RETROSPECTIVE.md`](../RETROSPECTIVE.md).

---

## The claim

Three separable statements, in increasing order of how much they should change
what you do:

1. **Quantization damages termination more than it damages reasoning.** A
   4-bit build fails to reach an answer on 24% of HumanEval problems where its
   bf16 parent finishes.
2. **Distilling on traces that terminate cleanly repairs most of it.** Cutting
   the truncation rate from 24% to 12% is where the bulk of a +3.0 point
   HumanEval gain comes from — 22 of the 28 problems gained were ones the
   stock model ran out of budget on.
3. **Any benchmark of a reasoning model that does not report its truncation
   rate is partly reporting its own token budget.** The same weights score
   82.3% or 88.4% on HumanEval depending only on whether the budget is 2,048
   or 4,096 tokens.

(3) is the one that generalises past this model.

---

## Evidence

### 1. The stock 4-bit model rambles, and the distilled one does not

Both models quantized with the identical oQ4 map (4-bit affine, group size 64,
120 modules promoted to 5/6/8 bits, 4.721 bits/weight average), measured by the
same harness at a 4,096-token HumanEval budget:

| | stock oQ4 | distilled oQ4 |
|---|---:|---:|
| HumanEval | 72.0% | **84.1%** |
| truncated before answering | **39 / 164 (24%)** | **19 / 164 (12%)** |
| mean generation | 1,721 tokens | **1,414 tokens** |

Item-level: the distilled model gained 28 problems and lost 8, and **22 of the
28 gains were problems where the stock model ran out of budget mid-reasoning**.
The gain is not mostly new capability; it is mostly finishing.

Truncation falls on every benchmark at once — MMLU 24 → 10, TruthfulQA 27 → 20,
HumanEval 39 → 19.

### 2. It is not a scoring artefact

The obvious objection is that this is a harness that punishes truncation, so of
course truncation explains the score. It reproduces on a harness that does not:
the oMLX server recovers an answer from output this harness scores as
unfinished, scoring the *stock* model 87.8% where this harness gives 72.0%.

On that forgiving harness, the same weights show up as **wall clock**:

| benchmark | generation time vs stock oQ4 |
|---|---:|
| HumanEval | **−33%** |
| MMLU | −15% |
| TruthfulQA | +26% |

A model that reaches its answer a third faster on the benchmark where it
improved, measured by a harness with no truncation penalty at all, is a
property of the model rather than of the scoring.

### 3. The effect is visible along the quantization ladder

Same weights, same harness, same protocol, different bit widths:

| build | MMLU | MMLU seconds | vs oQ4 |
|---|---:|---:|---:|
| oQ3 | 66.6% | 34,085 | **2.23×** |
| oQ4 | 78.0% | 15,262 | 1.00× |
| oQ8 | 83.1% | 16,434 | 1.08× |
| bf16 | 82.6% | 33,507 | 2.20× |
| **distilled oQ4** | **83.5%** | **13,007** | **0.85×** |

oQ3 is the interesting row. It has *fewer* bits than oQ4, so its per-token
throughput should be at least as high — yet it takes **2.23× as long** for the
same 1,000 questions while scoring 11.4 points worse. The extra time is extra
tokens: it is reasoning at much greater length and arriving somewhere worse.

bf16's 2.20× is a different effect (unquantized compute is simply slower per
token) and should not be read as rambling.

*Caveat:* these are wall-clock totals, not token counts. Separating
tokens-per-second from tokens-generated needs per-item logs the external
harness does not emit. The oQ3 row is strong but not airtight.

### 4. Multiple choice hides it

The repair shows up on code and barely on multiple choice — MMLU and
TruthfulQA moved within noise on the harness where HumanEval moved +12.2.

That asymmetry is itself evidence. A truncated *program* is worthless; a
truncated multiple-choice answer is often still recoverable from context, so
there is far less broken behaviour available to repair. **The benchmark that
most obviously shows this failure mode is the one whose output cannot be
salvaged.** Benchmarks that can salvage it will under-report the damage.

### 5. Budget sensitivity, on unchanged weights

The bf16 base, same weights, same harness, only the budget changed:

| budget | HumanEval | truncations |
|---|---:|---:|
| 2,048 | 82.3% | 20 |
| 4,096 | **88.4%** | 15 |

**Six points of "model quality" that are entirely an evaluation parameter.**
Note that 15 items still truncate at 4,096 — the number never reaches zero, it
just stops mattering as much.

---

## Counter-evidence, and a control

The account above would be more comfortable if every intervention moved
termination in the good direction. Two later runs are worth recording because
they did not.

**A run that made termination dramatically worse.** Training the same student
on the teacher's full next-token distribution rather than its sampled token
(`L = 0.3·CE + 0.7·KL` over the teacher's renormalised top-64) produced a model
that scored **77.2% MMLU — below the stock build — and took 69,897 seconds
against the distilled model's 13,007 on the same 1,000 questions, 5.4×.**

It did not become confused; it became unable to stop. The mechanism is what the
objective asks for: cross-entropy trains the student toward the single token the
teacher committed to, while KL trains it to reproduce the teacher's entire
distribution at every step — including the teacher's residual uncertainty about
whether to stop reasoning. A 35B teacher can carry that uncertainty and still
terminate. A 9B student at 4.72 bits cannot.

This is the same axis driven the wrong way, which is indirect support for the
mechanism — and a caution that "distill from a stronger teacher" is not
automatically safe for termination.

**The teacher does it too.** Rolling the distilled 9B over a fresh
4,974-prompt pool and sending only its 1,247 failures to the 35B teachers: the
code teacher verified on **29%** of those and **failed to terminate on 38%**
even after budget escalation to 4,608 tokens — against **73%** verified on
random prompts from the same pool under the same teacher and verifier. On the
problems a quantized 9B cannot finish, a 4-bit 35B frequently cannot finish
either. The failure is not specific to the small model; it is a property of
hard problems under a budget, and quantization moves the threshold.

**A control that changed nothing.** Doubling adapter capacity (rank 32 → 64,
byte-identical data, every other hyperparameter held) left accuracy flat within
one standard error *and* left wall clock normal: 13,319 s on MMLU against
13,007. A healthy model that is simply no better. That isolates the previous
result as specific to the KL objective rather than to any departure from the
baseline recipe.

---

## What follows from this

**For anyone benchmarking a reasoning model.** Report the truncation rate next
to the accuracy. Without it a score is a joint measurement of the model and the
budget, and the two are not separable after the fact. If a comparison involves
different models at a fixed budget, the one that reasons longer is being
penalised for something other than knowing less.

**For anyone quantizing a reasoning model.** Expect verbosity to rise as bits
fall, and expect it to cost more than the raw capability loss does. Measure
generation length before and after, not just accuracy — the cheapest early
warning is wall clock.

**For anyone distilling one.** Traces that terminate cleanly are the asset.
Selecting teacher traces by verified correctness gets this for free, since a
rambling trace usually fails its verifier. Objectives that transfer the
teacher's uncertainty transfer its hesitancy about stopping too, and a smaller
student may not be able to carry it.

**For anyone deploying one.** Repair it at decode time. A bias of
`0.02 × (n − 1024)` on the `</think>` logit while the block is open takes the
stock 4-bit build from 108 to 139 of 164 HumanEval problems at a 2,048-token
budget and the distilled build from 130 to 147, with no item that was correct
becoming wrong, and at 2,048 tokens the ramped model outscores the bf16 parent
at 4,096. A hard two-phase limit (close the block at `B − N`, allow `N` to
answer) recovers less (142 on the distilled build) and has a cliff to place.
Two attempts to train the same behaviour into the weights failed
(`DECISIONS.md` §46–49): self-distilling the harness-closed traces found too
few of them, and preference training against the model's own unterminated
traces churned without changing when it stops. The behaviour lives in the
quantized forward pass, and the repair belongs in the decoder.

**The honest limits of this.** One model family, one quantization scheme, one
hardware target. The mechanism behind the 4-bit build's verbosity is not
established here — this documents that it happens and that it dominates the
measured damage, not why a coarser weight grid produces longer reasoning.
The ramp result says the stop logit is depressed relative to continuing by an
amount that grows with position; it does not say which modules do it. Two
harnesses agreeing on direction while disagreeing by 15 points on the stock
model's absolute score is itself a reason to treat any single number here as
approximate.

---

## Reproducing

```bash
odistil eval --model student:oq4 --benchmark humaneval   # truncation count is
                                                         # reported next to the
                                                         # accuracy
```

```bash
odistil eval --model student:oq4 --benchmark humaneval --max-tokens 2048 --think-bias 1024:0.02   # soft HALT
odistil eval --model student:oq4 --benchmark humaneval --max-tokens 2048 --force-budget 256      # hard HALT
```

The runner records `truncated`, `tokens`, `seconds`, and for truncated items
the tail of the reasoning block, per item in
`runs/<tag>/eval/<benchmark>.jsonl`. Per-benchmark budgets live under
`eval.max_tokens` in the distill config; the finding in §5 above is reproduced
by moving `humaneval` between 2048 and 4096 and re-running the same weights.
