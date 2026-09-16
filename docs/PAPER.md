# Quantization Breaks Stopping Before It Breaks Reasoning: Repairing a 4-bit Reasoning Model by Distillation and by Decoding

*Draft — 16 September 2026. Numbers marked ⟨pending⟩ are being measured.*

## Abstract

Quantizing a 9B reasoning model to 4.7 bits costs it 4.6 points of MMLU and 3.7 of HumanEval against its bf16 parent. We show that most of this loss is not a loss of reasoning ability but of the ability to *stop*: the quantized model fails to reach an answer on 24% of HumanEval problems its parent finishes, and on a strict harness the entire measured gap is explained by truncation. Two very different interventions repair it. Sequence-level distillation from two 35B teachers, with rejection sampling on verified traces, produces a model at identical size and bit width that scores +5.5 MMLU and +3.0 HumanEval over the quantized baseline, beats the 1.8× larger 8-bit build on both, and matches its own 35B teacher on MMLU — while generating 15–33% fewer tokens. Budget forcing at decode time, with no training, recovers ⟨pending⟩ points of HumanEval on the same quantized weights, and a model forced to stop at 2,048 tokens outperforms the same model allowed 4,096. Eight further training runs with one controlled arm — varying data mixture, volume, length distribution, objective, filter, adapter capacity, and on-policy prompt selection — all fail to improve on the first, and the control shows that a second training pass costs 1–4 points regardless of its data. We conclude that for this model the recipe is not the lever, that termination is a decoding-time property before it is a weights property, and that any benchmark of a reasoning model that does not report its truncation rate is partly reporting its own token budget. Along the way we contribute a chunkwise-parallel training path for gated DeltaNet layers in MLX (24× memory, 135× speed at 4,096 tokens), now upstream.

---

## 1. Introduction

Post-training quantization is the default way to serve a reasoning model on consumer hardware, and its cost is usually reported as a few points of benchmark accuracy. This paper asks what those points *are*. On Ornith-1.5-9B — a hybrid-attention reasoning model with gated DeltaNet layers — the 4.7-bit build loses 4.6 MMLU and 3.7 HumanEval against bf16. We find that the loss is dominated by a single behaviour: the quantized model reasons past its token budget without reaching an answer far more often than its parent does.

That reframing matters because the two natural repairs — retrain, or change the decoder — address the same mechanism from different sides, and both work. It also matters for evaluation: a score obtained under a token budget is a joint measurement of the model and the budget, and the same weights move six points on HumanEval when the budget moves from 2,048 to 4,096 tokens.

Our contributions:

1. **A diagnosis.** Quantization damages termination more than reasoning (§5). Evidenced by truncation counts, item-level gains, cross-harness wall clock, the quantization ladder, a budget-sensitivity control on unchanged weights, and the asymmetry between code and multiple-choice.
2. **A training-side repair.** Two-teacher sequence-level distillation with rejection sampling (§3–4): +5.5 MMLU, +3.0 HumanEval at identical size and bit width; the resulting model beats the 8-bit build and matches its 35B teacher on MMLU.
3. **A decoding-side repair.** Budget forcing (§6) recovers ⟨pending⟩ of the same gap with no training, and stopping earlier than the model would stop itself improves accuracy.
4. **Eight controlled negative results** (§7), including a randomised control showing that continued training from the best checkpoint is itself harmful, and evidence that the 35B teacher does not express the calibrated uncertainty the student lacks.
5. **An engineering result** (§8): a differentiable chunkwise training path for gated DeltaNet in MLX, 24× memory and 135× speed at 4,096 tokens, contributed upstream.

Everything was measured on one Apple M4 Pro with 64 GB of unified memory.

---

## 2. Setup

**Student.** Ornith-1.5-9B, 32 layers, hybrid attention with gated DeltaNet (GDN) linear-attention layers, 248,044-token vocabulary, thinking-mode chat template. The deployed quantization is `oQ4`: 4-bit affine at group size 64 with 120 modules (predominantly linear-attention projections) promoted to 5, 6 or 8 bits, averaging 4.721 bits per weight. The promotion map is recoverable from the checkpoint's `config.json` and we replay it exactly, so every quantized comparison is like-for-like.

**Teachers.** Two 35B-A3B mixture-of-experts models at 4 bits, routed by measured strength:

| | MMLU | TruthfulQA | HumanEval | relative sampling speed |
|---|---:|---:|---:|---:|
| Qwen3.6-35B-A3B | 89.3 | 89.2 | 93.3 | 1.0× |
| Ornith-1.5-35B-A3B | 83.0 | 86.4 | 93.3 | 4.6× |

Qwen teaches knowledge and truthfulness; Ornith-35B teaches code, where it ties Qwen at a quarter of the cost. All three models share an index-identical vocabulary (verified: `vocab` and `merges` equal; Qwen's pre-tokenizer regex differs, which affects text encoding but not token identity), which is what makes logit-level distillation (§7.2) possible at all.

**Benchmarks and protocol.** MMLU (a fixed seeded 1,000-question sample of 14,042), TruthfulQA MC1 (all 817), HumanEval (all 164). Thinking enabled, greedy decoding. Two harnesses: the oMLX server, which is the deployment target and the source of all headline numbers, and our own (`odistil eval`), which records per-item truncation, token counts and wall clock. The harnesses disagree by up to 15 points on the *stock* model's HumanEval because oMLX recovers an answer from output ours scores as unfinished (§5.2); they agree on every direction. All models in a table are measured by the same harness under the same protocol.

**Evaluation hygiene.** TruthfulQA lists the correct choice first in all 817 items; presented in dataset order, every gold answer is "A" and a model that always picks A scores 100%. We shuffle choices under a per-item seed. Code is executed in a subprocess with a temporary working directory and a hard timeout. Training data is decontaminated against all three eval sets with 13-gram overlap.

**Hardware.** One Apple M4 Pro, 64 GB unified memory. A 9B full fine-tune does not fit (≈18 GB weights + ≈72 GB fp32 AdamW state); LoRA over the bf16 base does. Only one model is resident at a time.

---

## 3. Method

**Sequence-level distillation with rejection sampling.** For each prompt in a pool drawn from MMLU auxiliary-train, ARC, OpenBookQA, TriviaQA, SciQ and MBPP, the owning teacher generates one reasoning trace with thinking enabled. The trace is kept only if it verifies: the final letter matches gold (multiple choice), the normalised answer matches a gold alias (open QA), or the extracted program passes the dataset's own tests (code). Teacher accuracy on MMLU is 89.3%, so roughly one unfiltered knowledge trace in nine would be a confidently argued wrong answer. Traces that exceed the training length cap are dropped rather than truncated — a truncated target teaches the model to stop mid-answer.

**Train in bf16, quantize last.** LoRA (rank 32, top 16 of 32 layers, α = 16, dropout 0.05) over the bf16 student, 3,200 steps at batch size 1, lr 3×10⁻⁵ cosine with 150 warmup steps, 1,024-token cap. Adapters are fused into a bf16 checkpoint, which is then quantized with the replayed oQ4 map. Training on the quantized base (QLoRA) was measured and rejected: 49.0 GB peak against bf16's 47.3 and 26 s/iter against 10, because dequantisation every step costs more than the smaller weights save.

**Mixture.** 40% knowledge, 25% truthfulness, 35% code by sample count, rebalanced from whichever domain is most over-supplied so that no domain's scarcity discards another's data. v1 trained on 1,639 samples (33 held out).

---

## 4. Main result

All rows measured on oMLX under the protocol in §2.

| model | size | MMLU | TruthfulQA | HumanEval | HumanEval time |
|---|---:|---:|---:|---:|---:|
| Ornith-1.5-9B oQ3 | 3.9 GB | 66.6 | 67.6 | 72.0 | 5,585 s |
| Ornith-1.5-9B oQ4 (stock) | 4.9 GB | 78.0 | **80.7** | 87.8 | 4,171 s |
| Ornith-1.5-9B oQ8 | 8.9 GB | 83.1 | 80.7 | 88.4 | 4,797 s |
| Ornith-1.5-9B bf16 | 17 GB | 82.6 | 80.5 | **91.5** | 9,994 s |
| **distilled oQ4 (v1, this work)** | **4.9 GB** | **83.5** | 79.0 | 90.8 | **2,786 s** |
| Ornith-1.5-35B-A3B 4-bit (teacher) | 18 GB | 83.0 | 86.4 | 93.3 | 1,043 s |
| Qwen3.6-35B-A3B 4-bit (teacher) | 19 GB | **89.3** | **89.2** | 93.3 | 4,805 s |

Against the model it replaces, the distilled build is **+5.5 MMLU and +3.0 HumanEval at identical size and bit width**, −1.7 TruthfulQA. It beats the 1.8× larger oQ8 on both gains, edges the 3.5× larger bf16 parent on MMLU, and matches its own 35B code teacher on MMLU. It is also faster: −33% wall clock on HumanEval and −15% on MMLU against stock oQ4 — the same weights per token, fewer tokens per answer.

The TruthfulQA regression is real and is analysed in §7.4.

---

## 5. The mechanism: quantization breaks stopping

### 5.1 Truncation explains the gain

On our harness at a 4,096-token HumanEval budget, both models quantized with the identical map:

| | stock oQ4 | distilled oQ4 |
|---|---:|---:|
| HumanEval | 72.0% | **84.1%** |
| never reached an answer | **39 / 164 (24%)** | **19 / 164 (12%)** |
| mean generation | 1,721 tokens | **1,414 tokens** |

Item-level, the distilled model gained 28 problems and lost 8. **Twenty-two of the 28 gains were problems where the stock model ran out of budget mid-reasoning.** Truncation falls on every benchmark at once: MMLU 24 → 10, TruthfulQA 27 → 20, HumanEval 39 → 19.

### 5.2 It is not a scoring artefact

The oMLX harness recovers an answer from output ours scores as unfinished, scoring stock oQ4 at 87.8% where ours gives 72.0%. On that forgiving harness the same effect appears as **wall clock**: −33% on HumanEval and −15% on MMLU. A model that reaches its answer a third sooner on the benchmark where it improved, measured by a harness with no truncation penalty, is a property of the model rather than of the scoring.

### 5.3 The quantization ladder

Same weights, same harness, same 1,000 MMLU questions:

| build | MMLU | seconds | vs oQ4 |
|---|---:|---:|---:|
| oQ3 | 66.6 | 34,085 | **2.23×** |
| oQ4 | 78.0 | 15,262 | 1.00× |
| oQ8 | 83.1 | 16,434 | 1.08× |
| bf16 | 82.6 | 33,507 | 2.20× |
| distilled oQ4 | **83.5** | **13,007** | **0.85×** |

oQ3 has fewer bits than oQ4, so its per-token throughput should be at least as high; it takes 2.23× as long for the same questions while scoring 11.4 points lower. The extra time is extra tokens. (bf16's 2.20× is unquantized compute being slower per token, a different effect.) *Caveat:* these are wall-clock totals; the external harness does not emit per-item token counts.

### 5.4 Budget sensitivity on unchanged weights

The bf16 base, our harness, only the budget changed:

| budget | HumanEval | truncations |
|---|---:|---:|
| 2,048 | 82.3% | 20 |
| 4,096 | **88.4%** | 15 |

Six points of "model quality" that are entirely an evaluation parameter.

### 5.5 Multiple choice hides it

The repair shows up on code (+12.2 on our harness) and barely on multiple choice (MMLU and TruthfulQA within noise at n = 250). A truncated program is worthless; a truncated multiple-choice trace often still contains a recoverable letter. **The benchmark that most clearly shows this failure mode is the one whose output cannot be salvaged.** Benchmarks that can salvage it will under-report the damage.

### 5.6 The teacher does it too

Rolling the distilled model over a fresh 4,974-prompt pool and sending only its 1,247 failures to the teachers (§7.6): the 35B code teacher verified on **29%** of those and **failed to terminate on 38%** even after budget escalation to 4,608 tokens — against 73% verified on random prompts from the same pool. The failure is not specific to the small model; it is a property of hard problems under a budget, and quantization moves the threshold.

---

## 6. Repairing it at decode time: budget forcing

If the failure is stopping, the cheapest test is to stop the model for it. Following s1 [Muennighoff et al., 2025], when the reasoning block has not closed by the time the budget is nearly spent, we close it — appending a short commit phrase and `</think>` — and allow a reserved number of tokens for the answer. Implementation detail that mattered: the reserve is a second phase for *anything* that hit the first-phase cap, with `</think>` forced only for items still reasoning; an earlier version that took the reserve from every item's budget cut a correct solution mid-answer.

All rows: our harness, v1 unless stated, reserve N = 256.

| model | budget | plain | truncated | forced | forced → correct | Δ |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 2,048 | 130 / 164 (79.3%) | 24 | 142 / 164 (**86.6%**) | 12 of 29 | **+12** |
| v1 | 4,096 | 138 / 164 (84.2%) | 19 | ⟨pending⟩ | ⟨pending⟩ | ⟨pending⟩ |
| stock oQ4 | 4,096 | ⟨pending⟩ | ⟨pending⟩ | ⟨pending⟩ | ⟨pending⟩ | ⟨pending⟩ |
| v1, MMLU ×250 | 3,072 | ⟨pending⟩ | ⟨pending⟩ | ⟨pending⟩ | ⟨pending⟩ | ⟨pending⟩ |

Two observations already stand. First, **v1 forced at 2,048 tokens (86.6%) outperforms v1 allowed 4,096 (84.2%)** — half the budget, more correct answers. Second, in a 24-item pilot, one problem the model reasoned on to ~2,050 tokens and answered *wrong* was answered *right* when cut at 1,792: past some point, continued reasoning has negative value and the model does not detect it. Wall clock is unchanged by forcing.

---

## 7. What did not work, and what the controls established

Eight further runs, each moving one variable from v1, each with a prediction recorded before measurement. All measured on oMLX.

| run | change | MMLU | TruthfulQA | HumanEval |
|---|---|---:|---:|---:|
| **v1** | — | **83.5** | 79.0 | **90.8** |
| v2 | misconception slice replaces TriviaQA; brief code | 83.1 | 76.9 | 85.4 |
| v3 | abstention slice added; v1's code traces restored | 83.6 | 77.4 | 87.2 |
| v4 | 1,024-token length bias removed, all else held | 83.2 | 76.6 | 89.6 |
| v5 | objective → 0.3·CE + 0.7·KL(teacher‖student), top-64 | 77.2 | — | — |
| v6 | filter → recover verified abstentions | stopped: 22 usable, 0 under cap | | |
| v7 | adapter rank 32 → 64, data byte-identical | 83.2 | 77.6 | 88.4 |
| v8 | on-policy: train on v1's own failures, continued from v1 | 81.6 | 76.3 | 85.4 |
| v8-control | same, random prompts | 83.2 | 77.7 | 86.6 |

Across the six completed comparable runs, **21 of 24 benchmark deltas against v1 are negative and none is positive beyond noise.**

### 7.1 Data composition (v2–v4)

MMLU sat at 83.5, 83.1, 83.6, 83.2 across mixtures that differed in source, volume (1,672 → 4,075 samples) and length distribution; v1 already matches its 35B teacher there. v4 is the cleanest of the three: v1 had trained only on traces under 1,024 tokens (1,758 of median 678; 1,735 of median 1,449 never seen), which §5 predicts should matter. Lifting the cap to 2,048 with sample count, steps and every hyperparameter held — a `max_samples` cap added so the length change could not smuggle in volume — moved nothing. The pre-registered prediction (HumanEval improves most) was wrong.

### 7.2 Objective: forward KL makes the model unable to stop (v5)

Training against the teacher's renormalised top-64 distribution (99.93% of its mass, measured) instead of its sampled token produced the first model *below stock*: 77.2 MMLU in **69,897 s against v1's 13,007 s** on the same questions — 5.4×. It did not become confused; it became unable to stop. This is the mode-covering failure MiniLLM [Gu et al.] describes: a 9B at 4.7 bits cannot cover a 35B's low-probability modes, including its residual uncertainty about whether to continue reasoning. Three failure modes were ruled out before accepting the result (vocabulary identity across teachers, top-k coverage, span alignment — all clean); a latent bug in the batcher was found and fixed on the way. v7 (§7.4) later confirmed the collapse is specific to the objective: its wall clock was normal.

### 7.3 Adapter capacity (v7)

Every earlier run used rank 32 over the top 16 layers, chosen once and never revisited. Doubling to 86.6M trainable parameters on byte-identical data (sha256-verified) changed nothing: all three deltas inside one standard error, wall clock normal. Rank 64 across all 32 layers does not fit in 64 GB — backprop *depth*, not rank, is what costs. This converts "capacity, not recipe" from an inference reached by eliminating recipes into a measurement.

### 7.4 The TruthfulQA regression is an abstention deficit, and the teacher cannot supply the fix (v3, v6)

Per-item, v1 matches stock on ordinary TruthfulQA questions (81.3 vs 81.4) and **loses 11 points on the 117 whose correct answer expresses uncertainty** (65.0 vs 76.1; 19 lost, 6 gained, p ≈ 0.016). Rejection sampling keeps only confident correct answers, so the student never sees warranted uncertainty. Adding 527 SQuAD-v2 unanswerable items (v3) moved TruthfulQA *down* 1.6 — reading-comprehension abstention is a different skill. Changing the filter instead (v6) — re-asking the teacher, with its confidence made explicit, the 122 open-ended prompts it had answered wrong, and keeping only the traces where it declined — yielded 22 usable traces, of which **0** fit the training cap. The number that matters is the other one: **on questions it had already answered wrong, invited to say it did not know, Qwen3.6-35B re-asserted a confident answer 82% of the time.** Sequence-level distillation can only transmit what the teacher emits, and this teacher does not emit calibrated uncertainty.

### 7.5 A second training pass from the best checkpoint is itself harmful (v8 and its control)

v8 tested on-policy prompt selection: roll v1 out over a fresh pool, keep the 1,247 prompts it fails, have the teacher answer only those, train on the 330 that verify, continuing from v1. A control arm trained on 330 *random* prompts from the same pool with everything else identical. **Both arms lost to v1 on all three benchmarks, by similar amounts.** Since the arms differ only in prompts, the second pass is the damage, not the selection. v1 is a sharp optimum. Every method that starts from v1 must first beat a 1–4 point tax; every method that starts from the base must re-earn v1's gains.

### 7.6 Yield asymmetry

The same teacher and verifiers passed **73% of random prompts and 26% of v1's failures**. The prompts a 9B fails are largely ones a 35B struggles to finish too (§5.6).

---

## 8. Engineering: a chunkwise training path for gated DeltaNet

Training was pinned at 1,024-token sequences, which rejected 46% of every trace generated. The cause was not the hardware: `mlx_lm` trains GDN layers through a sequential Python loop over every timestep — its own docstring calls it a reference implementation for prefill — because the inference Metal kernel has no gradient. Measured: ≈8 MB of autograd state per token per layer.

The gated delta rule admits a chunkwise-parallel form [Yang et al., 2024a,b] keeping O(T/C) states instead of O(T). Implemented in MLX with two details that mattered: `mx.linalg.tri_inv` is CPU-only with no VJP, so the unit-triangular inverse is built from matmuls by 2×2 block recursion in log₂C levels; and the textbook derivation divides by the cumulative decay, which underflows on strongly decaying heads — carrying bounded ratios G_j/G_i ≤ 1 instead is what makes it work on a real model. A fused Metal triangular *solve* with a hand-written VJP replaces the inverse (used exactly once as `inv @ rhs`), adding 2.1–2.7×.

One GDN layer, forward + backward, cumulative against stock `mlx_lm` at 4,096 tokens: **108.33 GB / 39.02 s → 4.60 GB / 0.29 s — 24× memory, 135× speed.** Outputs match the sequential reference to ~3×10⁻⁷ relative in fp32 across all five gradients; the inference kernel matches at 100% top-1 on the full 9B. End-to-end training throughput doubled (55 → 111 tokens/s); 2,048-token training fits in 41.8 GB. Contributed as [mlx-lm#1870](https://github.com/ml-explore/mlx-lm/pull/1870) (chunkwise path, five model families) with the fused solve queued behind it.

---

## 9. Related work

**Distillation objectives.** MiniLLM [Gu et al., 2023] argues forward KL is mode-covering and ill-suited to generative distillation; GKD [Agarwal et al., 2023] adds on-policy sampling and generalised divergences; DistiLLM [Ko et al., 2024] proposes skew KL; TAID [Shing et al., 2025] interpolates the target toward the teacher over training. Our v5 is a direct instance of the forward-KL failure these predict, with termination as the specific casualty. **Test-time compute.** s1 [Muennighoff et al., 2025] introduces budget forcing; our §6 applies it to a quantized model and finds that forcing earlier than the model would stop improves accuracy. **Quantization-aware training.** Recover-LoRA [2025], EfficientQAT [Chen et al., 2024], LoTA-QAF [2025] train through the quantizer; we did not, and §7.5's finding that continued training from the best checkpoint is harmful bears on whether they could help here. **Linear attention training.** Yang et al. [2024a,b] give the chunkwise delta-rule form our §8 implements. **Byte-level distillation.** Marathe et al. [2026] convert token logits to byte logits for cross-tokenizer distillation; their objective is forward KL and their setting is pretraining from scratch, where the failure in §7.2 does not arise.

---

## 10. Limitations

One model family, one quantization scheme, one hardware target. The *mechanism* behind the quantized model's verbosity is documented, not explained: we show that a coarser weight grid produces longer reasoning and that this dominates the measured damage, not why. The quantization-ladder timing (§5.3) is wall clock, not token counts. Two harnesses disagree by 15 points on the stock model's absolute HumanEval while agreeing on every direction; any single absolute number here should be treated as approximate. TruthfulQA has no verifier and its regression was not repaired. The negative results are specific to sequence-level KD under LoRA on 64 GB; token-level on-policy distillation with a mode-seeking divergence and termination-token masking, quantization-aware self-distillation, and full-parameter updates were not run, and §7.5 constrains all of them.

---

## 11. Conclusion

A 4-bit reasoning model's dominant failure is that it does not stop. Distilling on traces that terminate cleanly repairs most of it; so does closing the reasoning block at the harness. Nothing else we tried — and we tried the obvious things carefully, with predictions on the record — improved on the first repair, and the one randomised control we ran shows that trying harder from the best checkpoint makes it worse. The practical recommendations are short: report truncation rates next to accuracy; measure generation length before and after quantizing; and treat the token budget as part of the model being evaluated.

---

## Reproducibility

Code, configs, every run's adapters, fused checkpoints, teacher traces, per-item evaluation records and the 45-section decision log are at `adityak74/ornith-1.5-9b-distil`. `benchmarks/baseline.json` holds every number in this paper with its measurement date and harness. `odistil eval --force-budget N` reproduces §6.

## References

- Agarwal, R. et al. (2023). On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. arXiv:2306.13649.
- Chen, M. et al. (2024). EfficientQAT. arXiv:2407.11062.
- Gu, Y. et al. (2023). MiniLLM: Knowledge Distillation of Large Language Models. arXiv:2306.08543.
- Ko, J. et al. (2024). DistiLLM. arXiv:2402.03898.
- Marathe, K. et al. (2026). Breaking the Token Ceiling: Distilling Smaller, Stronger Byte Models. arXiv:2609.12303.
- Muennighoff, N. et al. (2025). s1: Simple test-time scaling. arXiv:2501.19393.
- Recover-LoRA (2025). arXiv:2510.08600.
- Shing, M. et al. (2025). TAID. arXiv:2501.16937.
- Yang, S. et al. (2024a). Parallelizing Linear Transformers with the Delta Rule over Sequence Length. arXiv:2406.06484.
- Yang, S., Kautz, J., Hatamizadeh, A. (2024b). Gated Delta Networks. arXiv:2412.06464.
