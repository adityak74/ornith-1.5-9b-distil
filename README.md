# ornith-1.5-9b-distil

Two-teacher distillation of **Ornith-1.5-9B** on Apple Silicon / MLX.

- **Qwen3.6-35B-A3B-4bit** teaches knowledge and truthfulness (89.3 MMLU / 89.2 TruthfulQA).
- **Ornith-1.5-35B-A3B-4bit** teaches code — it ties Qwen on HumanEval (93.3) and
  finishes the benchmark 4.6× faster, so it does the bulk of the generation.

Goal: take the shipped oQ4 student from **78.0 / 80.7 / 87.8** to
**86–89 / 86–89 / 92–94** on MMLU / TruthfulQA / HumanEval, at ~5–6 GB.

Full reasoning, targets and open questions: [`docs/PLAN.md`](docs/PLAN.md).
Measured baselines: [`benchmarks/baseline.json`](benchmarks/baseline.json).

## Setup

```bash
make setup   # uv sync (python 3.12, mlx, mlx-lm, datasets)
make check   # volume mounted? models present? tokenizers compatible?
```

All six checkpoints are already on the external volume `/Volumes/SATECHI/.omlx`
(the oMLX model dir), wired into `configs/models.yaml` — nothing to download,
but the volume has to be mounted. `runs/` stays on the internal disk.

## Pipeline

Every stage is resumable — re-running skips ids already on disk — and writes
under `runs/<out_dir>/` from `configs/distill.yaml`.

| Stage | Command | Output |
|---|---|---|
| 1. prompt pool | `odistil prompts` | `runs/v1/prompts.jsonl` |
| 2. teacher traces | `odistil teach` | `runs/v1/teacher/{knowledge,code}.jsonl` |
| 3. verify + mix | `odistil dataset` | `runs/v1/train/{train,valid}.jsonl` |
| 4. LoRA distill | `odistil train` | `runs/v1/adapters/` |
| 5. fuse | `odistil fuse` | `runs/v1/fused/` (bf16) |
| 6. quantize | `odistil quantize --variant q4` | `runs/v1/quant/q4/` |
| 7. evaluate | `odistil eval --model runs/v1/fused` | `runs/v1/eval/<tag>/` |
| 8. compare | `odistil report --time` | `runs/v1/report.txt` |

Stage 3 is where the quality comes from: a teacher trace is kept only if its
answer is verifiably correct (gold letter for MCQ, gold alias for open QA,
passing unit tests for code), and any prompt sharing a 13-gram with MMLU test,
TruthfulQA or HumanEval is dropped.

## Smoke test

`make smoke` runs the whole thing on 24 items and 20 training iterations —
it needs the models downloaded, but finishes in minutes.

## Evaluation

The harness reproduces the baseline protocol: MMLU sampled at 1000/14042 with a
fixed seed, TruthfulQA MC1 in full (817), HumanEval in full (164), thinking on,
greedy decoding, and generated code executed in a sandboxed subprocess.

```bash
odistil eval --model student:oq4  --tag baseline-oq4   # re-measure the shipped model
odistil eval --model runs/v1/fused --tag distilled-bf16
odistil report --time
```

```
Model                                  MMLU    TRUTHFULQA     HUMANEVAL
-----------------------------------------------------------------------
Ornith-1.5-9B-MLX-oQ4                 78.0%         80.7%         87.8%
Ornith-1.5-9B-MLX                     82.6%         80.5%         91.5%
Qwen3.6-35B-A3B-MLX-4bit              89.3%         89.2%         93.3%
Ornith-1.5-35B-A3B-MLX-4bit           83.0%         86.4%         93.3%
target (low)                          86.0%         86.0%         92.0%
target (high)                         89.0%         89.0%         94.0%
```

## Layout

```
configs/     models.yaml (registry) · distill.yaml (pipeline) · smoke.yaml
src/odistil/ cli.py check.py config.py mlxutil.py codeexec.py textnorm.py
             pipeline/{prompts,teach,dataset,train}.py
             eval/{tasks,runner,report}.py
benchmarks/  baseline.json — the measured numbers this project is judged against
runs/        generated; gitignored
```

## Notes

- Training is rank-64 LoRA over the **bf16** student, not the quantized one: a
  full fine-tune of a 9B needs ~90 GB with AdamW and will not fit in 64 GB.
  Distilling in full precision and quantizing afterwards also recovers part of
  the 4.6-point MMLU loss that bf16 → oQ4 currently costs.
- All three models share one 248,044-token vocabulary, so top-k logit
  distillation is possible as a v2 (see `docs/PLAN.md`); v1 is sequence-level
  with rejection sampling.
- `odistil quantize --variant oq4` reproduces the shipped oQ mixed-bit map
  (4-bit affine g64 with 120 modules promoted to 5/6/8 bits, mostly
  `linear_attn` projections) by reading it out of the reference checkpoint, so
  the final number is comparable to the shipped oQ4 rather than to a generic q4.
  Set `quantize.recipes.oq4` to a command if the real quantizer does more than
  choose bit widths.
