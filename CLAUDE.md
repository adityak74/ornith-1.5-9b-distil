# CLAUDE.md

Distillation pipeline for Ornith-1.5-9B on MLX. Read `docs/PLAN.md` first — it
holds the reasoning behind the two-teacher split and the targets.

## Working here

- `uv run odistil <stage>` for everything; `make` targets wrap the common ones.
- `make test` (pytest) and `make lint` (ruff) must pass before a commit.
- Iterate with `--distill-config configs/smoke.yaml`, which is sized for a
  minutes-long full pass. Never smoke-test against `configs/distill.yaml`.
- Every stage is resumable and append-only. Preserve that: new stages must skip
  ids already present in their output file rather than truncating it.

## Invariants

- **Never train on eval data.** MMLU test, TruthfulQA (all 817) and HumanEval
  are eval-only. `dataset` decontaminates with 13-grams; `--skip-decontam` is
  for smoke tests only and must never be used for a real run.
- **Rejection sampling is not optional.** Unverified teacher output only enters
  the training set for `code_free` prompts, which have no verifier.
- Model-generated code is untrusted: it runs only through
  `codeexec.run_program` (subprocess, temp cwd, hard timeout).
- Baseline numbers in `benchmarks/baseline.json` were measured externally.
  Don't edit them to match a new run — add the new run as its own eval tag.

## Hardware

Apple M4 Pro, 64 GB unified. A 9B full fine-tune does not fit; LoRA over the
bf16 base does. Only one model is resident at a time — `mlxutil.load` is
LRU-cached, so interleaving two teachers in one process will thrash.
