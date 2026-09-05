# Distillation plan — Ornith-1.5-9B

## Where we are

Measured on an M4 Pro / 64 GB, thinking on, MMLU sampled at 1000/14042,
TruthfulQA and HumanEval full (`benchmarks/baseline.json`):

| Model | MMLU | TruthfulQA | HumanEval | total bench time |
|---|---:|---:|---:|---:|
| Ornith-1.5-9B (bf16) | 82.6 | 80.5 | **91.5** | 18.4 h |
| Ornith-1.5-9B-oQ8 | 83.1 | 80.7 | 88.4 | 8.3 h |
| Ornith-1.5-9B-oQ4 | 78.0 | 80.7 | 87.8 | 8.0 h |
| Ornith-1.5-9B-oQ3 | 66.6 | 67.6 | 72.0 | 16.6 h |
| Qwen3.6-35B-A3B-4bit | **89.3** | **89.2** | 93.3 | 6.1 h |
| Ornith-1.5-35B-A3B-4bit | 83.0 | 86.4 | **93.3** | **2.6 h** |

Two facts drive everything below.

1. **The teachers are complementary, not ranked.** Qwen is +6.3 MMLU and +2.8
   TruthfulQA over Ornith-35B, but they tie exactly on HumanEval (153/164) —
   and Ornith-35B gets there 4.6× faster. Qwen teaches knowledge and
   truthfulness; Ornith-35B teaches code, and does the bulk of the generation
   work because it is cheap.
2. **oQ4 is losing capability the 9B already has.** bf16 → oQ4 costs 4.6 pts of
   MMLU and 3.7 pts of HumanEval. Part of the target gap is a quantization
   problem, not a knowledge problem, so the pipeline distills in full precision
   and only then quantizes, with a quantization A/B at the end.

## Targets

| | now (oQ4) | target |
|---|---:|---:|
| MMLU | 78.0 | 86–89 |
| TruthfulQA | 80.7 | 86–89 |
| HumanEval | 87.8 | 92–94 |

## Method

Sequence-level (dark-knowledge-free) distillation with rejection sampling:

```
                  Qwen3.6-35B-A3B-4bit          Ornith-1.5-35B-A3B-4bit
                  knowledge · truthfulness      code · agentic · tools
                            │                             │
                     traces + answers             traces + answers
                            └───────────┬─────────────────┘
                               verify (gold letter / alias / unit tests)
                               decontaminate (13-gram vs eval sets)
                               mix 40 / 25 / 35
                                        │
                            LoRA r=64 over Ornith-1.5-9B bf16
                                        │
                                     fuse → bf16
                                        │
                               quantize → q4 / oQ4 + recovery
                                        │
                            MMLU · TruthfulQA · HumanEval
```

**Sequence-level now, logit distillation next.** KL against teacher logits
needs a shared tokenizer, and it turns out we have one: `Ornith-1.5-9B`,
`Ornith-1.5-35B-A3B-4bit` and `Qwen3.6-35B-A3B-4bit` all carry an *identical*
248,044-token vocabulary with an identical token→id mapping (the two Ornith
`tokenizer.json` files are byte-identical; Qwen's differs only in file
formatting). Both teachers are architecturally `qwen3_5(_moe)`.

That makes true distillation viable, and v2 should do it — offline, not online:
during stage 2, record the teacher's top-64 logprobs per generated token
alongside the text, then train the student on a mixture of cross-entropy and
KL against those sparse targets. Offline avoids holding a 4-bit 35B teacher
(~19 GB) and the bf16 student (~18 GB) plus optimizer state in a 52 GB working
set at once, and lets the same trace be reused across training runs. It costs
disk (~64 floats + ids per token) and nothing else, so stage 2 is the place to
add it before spending the Qwen generation hours.

v1 stays sequence-level because the rejection-sampling filter is what buys most
of the gain, and it is the part that has to be right first.

**Why LoRA, not full fine-tuning.** A 9B in bf16 is ~18 GB of weights; AdamW
adds ~72 GB of fp32 state. That does not fit in 64 GB of unified memory.
Rank-64 LoRA on all layers over the *full-precision* base, then fused, is the
largest thing that fits — and it is applied to the bf16 model precisely so the
quantization step starts from an improved full-precision checkpoint.

**Why rejection sampling.** Teacher traces are only kept when the final answer
is verifiably correct: the letter matches gold for MCQ, the answer contains a
gold alias for open QA, and generated code actually passes the source dataset's
tests. Qwen is 89.3% on MMLU, so ~11% of unfiltered knowledge traces would be
confidently-argued wrong answers — exactly the thing a student overfits to.

**Decontamination.** MMLU test, TruthfulQA (all 817 — it has no train split, so
truthfulness is taught indirectly via TriviaQA and SciQ plus abstention-style
prompting) and HumanEval are indexed as 13-grams; any training prompt sharing
one is dropped before writing `train.jsonl`.

## What the checkpoints actually are (verified on disk)

Everything lives on `/Volumes/SATECHI/.omlx` (the oMLX model dir from
`~/.omlx/settings.json`), wired into `configs/models.yaml`:

- `ornith-ai/Ornith-1.5-9B-MLX` — unquantized **bf16**, hidden 4096,
  `model_type: qwen3_5`, 17 GB. Trainable as-is.
- Both teachers present at 4-bit, 18–19 GB each.
- `Ornith-1.5-9B-MLX-oQ{8,4,3,2}` — the shipped students, not on the Hub.

**oQ is a mixed-bit affine scheme, and the map is recoverable.** oQ4 is 4-bit
affine at group size 64 with 120 individual modules promoted: 110 to 5 bits,
7 to 6, 3 to 8 — concentrated in the `linear_attn` projections (`out_proj`,
`in_proj_{a,b,z}`), plus some `mlp.down_proj` and attention heads. oQ3 and oQ2
use the same shape at a lower base. `odistil quantize --variant oq4` reads that
map straight out of the reference checkpoint's `config.json` and replays it on
the distilled model via `mlx_lm.convert(quant_predicate=...)`, so the final
comparison against the shipped oQ4 is apples-to-apples rather than
"our q4 vs their oQ4".

That the promoted modules cluster on `linear_attn` is also a hint about where
the 4.6-point bf16→oQ4 MMLU loss lives — a v2 quantization experiment can push
those specific modules higher and measure MMLU alone.

## Throughput note

The harness batches generation (8 prompts at a time), which is much faster than
the baseline runs: 16 HumanEval items on oQ4 took 98 s (~6 s/item) against the
baseline's ~25 s/item. Full-suite estimate per model is roughly 1.5–2 h rather
than 8 h, so re-measuring every variant through this harness is affordable.

## Order of work

1. `odistil check` — confirms the external volume is mounted and every model is
   present (nothing to download; all six checkpoints are already on disk).
2. Re-run the oQ4 baseline through *this* harness (`make eval-baseline`) so the
   numbers above and the post-distillation numbers are same-harness comparable.
3. Code slice first: it is the cheapest teacher and the target closest to reach.
   `odistil teach --teacher code` → dataset → a short LoRA → HumanEval only.
   If HumanEval on the fused bf16 does not clear 91.5 (the undistilled bf16
   score), stop and fix the recipe before spending Qwen hours.
4. Knowledge/truthfulness slice on Qwen (the long pole: ~9k prompts).
5. Full train, fuse, eval bf16.
6. Quantize and A/B: mlx q4/q8 vs the oQ recipes, plus
   `mlx_lm convert --quant-predicate mixed_3_6` as a recovery option — the oQ4
   MMLU drop suggests some layers want more bits than others.

## Open questions

- ~~What produces the oQ artifacts?~~ The per-module bit map is reproducible
  from the reference configs (above). If the real oQ pipeline does more than
  pick bit widths — calibration, error feedback, activation-aware scaling —
  wire that command into `quantize.recipes` so stage 6 matches it exactly.
- ~~Is the 9B on the Hub trainable?~~ Confirmed: `ornith-ai/Ornith-1.5-9B-MLX`
  is unquantized bf16, hidden size 4096. Good to train against.
- MMLU truncation: ~1 in 8 sanity items hit the 2048-token cap before
  answering. Worth measuring the truncation rate on the full oQ4 baseline run
  before assuming the 78.0 → 86+ gap is all knowledge.
- The external volume has only 67 GB free; `runs/` (fused bf16 ≈ 17 GB plus
  quants) stays on the internal disk, which has 251 GB.
