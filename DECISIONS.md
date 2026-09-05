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
