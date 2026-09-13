"""Stage 2b -- recover the abstentions that rejection sampling throws away.

v1 matches the stock model exactly on ordinary TruthfulQA questions (81.3% vs
81.4%) and loses 11 points on questions whose correct answer is "I don't know"
(65.0% vs 76.1%, p ~ 0.016). The cause is the filter, not the mixture: every
training target is a *confident correct answer*, because that is the only thing
rejection sampling keeps. The student never sees warranted uncertainty.

v3 attacked this by adding a slice -- 527 SQuAD-v2 unanswerable items -- and
lost 1.6 points. That is a different skill: "the passage does not say" is
reading comprehension, while TruthfulQA hedging is "no good evidence supports
this". Adding out-of-domain abstention data did not transfer.

This stage changes the filter instead. When the knowledge teacher answers an
open-ended question *wrong*, that is, by construction, a question on which
confident assertion was not warranted -- and v1 simply discarded it. Here those
prompts are put back to the teacher with its confidence made explicit. If it
then declines, the trace is kept, paired with the **original** prompt.

Two honesty constraints, both deliberate:

- The target is teacher-generated and verified, never fabricated. A re-ask that
  produces another confident answer is dropped, exactly like any other trace
  that fails its verifier. This is rejection sampling with the gold inverted,
  not a synthetic label.
- **MCQ failures are excluded on purpose.** 135 of them exist, and MMLU imposes
  no penalty for guessing, so teaching abstention on multiple choice would put
  the one metric v1 wins at risk to chase the one it loses. Open-ended only.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import Config
from ..mlxutil import generate_batch
from .dataset import _verify
from .teach import SYSTEM

# The re-ask. It does not tell the teacher to decline -- that would manufacture
# the answer we are looking for and the verifier below would be a rubber stamp.
# It asks for a confidence judgement and leaves both outcomes open, so a teacher
# that does know the fact says so and the sample is dropped.
CALIB_TMPL = (
    "{question}\n\n"
    "Before answering, judge honestly whether you actually know this. "
    "If you do, answer it. If you are relying on a guess, a half-remembered "
    "detail, or a plausible-sounding association, say plainly that you do not "
    "know rather than presenting a guess as fact.\n"
    "End your reply with a final line of exactly 'Answer: <answer>' or "
    "'Answer: I don't know'."
)

_DECLINES = (
    "i don't know", "i do not know", "don't know", "do not know",
    "not sure", "unsure", "cannot say", "can't say", "no reliable",
    "not confident", "unable to say", "i'm not certain", "i am not certain",
)


def declines(answer: str) -> bool:
    """True when the teacher's final line is an admission rather than a fact."""
    from ..textnorm import final_answer

    said = (final_answer(answer) or answer).strip().lower()
    return any(w in said for w in _DECLINES)


def _failed_qa(cfg: Config) -> list[dict]:
    """Open-ended traces the verifier rejected -- the discarded pool."""
    dcfg = cfg.distill["dataset"]
    out = []
    for path in sorted((cfg.out_dir / "teacher").glob("*.jsonl")):
        if path.stem == "hedge":
            continue
        with path.open() as f:
            for line in f:
                rec = json.loads(line)
                if rec["kind"] != "qa":
                    continue
                ok, _ = _verify(rec, dcfg)
                if not ok:
                    out.append(rec)
    return out


def run(cfg: Config, limit: int | None = None) -> Path:
    """Re-ask the failures, keep the ones the teacher declines."""
    hcfg = cfg.distill.get("hedge") or {}
    out_path = cfg.path("teacher", "hedge.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        with out_path.open() as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}

    pool = [r for r in _failed_qa(cfg) if f"hedge-{r['id']}" not in done]
    if limit:
        pool = pool[:limit]
    print(f"[hedge] {len(pool)} rejected open-ended traces to re-ask "
          f"({len(done)} already done)")
    if not pool:
        return out_path

    teacher = cfg.teacher_for("truthfulness")
    tconf = cfg.distill["teach"]
    bs = tconf["batch_size"]
    base_budget = hcfg.get("max_tokens") or teacher["max_tokens"]
    retries = tconf.get("max_retries", 2)
    kept = confident = truncated = 0

    def _ask(items, budget):
        return generate_batch(
            teacher["path"],
            [CALIB_TMPL.format(question=_question_of(r)) for r in items],
            system=SYSTEM,
            max_tokens=budget,
            temp=teacher["temp"],
            top_p=teacher["top_p"],
            think=tconf["think"],
            batch_size=bs,
        )

    with out_path.open("a") as f:
        for i in range(0, len(pool), bs):
            chunk = pool[i : i + bs]
            comps = _ask(chunk, base_budget)

            # A re-ask that ran out of budget inside <think> has no answer at
            # all, and an empty answer is not a confident one -- scoring it as
            # "still confident" silently discarded every sample on the first
            # attempt. Same escalation the teach stage uses.
            for attempt in range(retries):
                stuck = [j for j, c in enumerate(comps) if not (c.text or "").strip()]
                if not stuck:
                    break
                budget = int(base_budget * 1.5 ** (attempt + 1))
                print(f"  retrying {len(stuck)} truncated at {budget} tokens")
                redone = _ask([chunk[j] for j in stuck], budget)
                for j, c in zip(stuck, redone, strict=True):
                    comps[j] = c

            for rec, comp in zip(chunk, comps, strict=True):
                if not (comp.text or "").strip():
                    truncated += 1
                    continue
                if not declines(comp.text):
                    confident += 1
                    continue
                f.write(json.dumps({
                    "id": f"hedge-{rec['id']}",
                    "domain": rec["domain"],
                    "kind": "hedge",
                    # The ORIGINAL prompt: the student must learn to hedge when
                    # asked normally, not only when asked about its confidence.
                    "prompt": rec["prompt"],
                    "gold": rec["gold"],
                    "answer": comp.text,
                    "think": comp.think,
                    "meta": {**rec.get("meta", {}), "via": "calibration re-ask"},
                }) + "\n")
                f.flush()
                kept += 1
            print(f"  {min(i + bs, len(pool))}/{len(pool)}  kept {kept}  "
                  f"confident {confident}  no-answer {truncated}",
                  end="\r", flush=True)

    print(f"\n[hedge] kept {kept}; dropped {confident} that answered confidently "
          f"and {truncated} that produced no answer within budget "
          f"-> {out_path.name}")
    return out_path


def _question_of(rec: dict) -> str:
    """The bare question, recovered from the QA template the prompt was built with."""
    return rec["prompt"].split("\n\nAnswer accurately")[0].strip()
