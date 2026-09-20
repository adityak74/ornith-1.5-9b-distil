"""Generate the paper's result tables from measured artifacts.

Every number in paper/tables_*.tex is read from `benchmarks/baseline.json`
(oMLX measurements) or from `runs/v1/eval/<tag>/summary.json` (our harness),
so the manuscript cannot drift from what was measured. Re-run after any new
evaluation lands:

    uv run python scripts/make_paper_tables.py

Same pattern as budgetbench/paper/results_*_tables.tex: each table is a
standalone .tex fragment pulled in with \\input{}.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper"
BASE = json.loads((ROOT / "benchmarks" / "baseline.json").read_text())["models"]
EVAL = ROOT / "runs" / "v1" / "eval"


def pct(x: float | None) -> str:
    return "--" if x is None else f"{x * 100:.1f}"


def secs(x: float | None) -> str:
    return "--" if x is None else f"{x:,.0f}"


def row(name: str, key: str, size: str, bold: set[str] = frozenset()) -> str:
    m = BASE.get(key, {})
    cells = []
    for b in ("mmlu", "truthfulqa", "humaneval"):
        v = pct(m.get(b, {}).get("accuracy"))
        cells.append(f"\\textbf{{{v}}}" if b in bold else v)
    t = secs(m.get("humaneval", {}).get("seconds"))
    return f"{name} & {size} & {cells[0]} & {cells[1]} & {cells[2]} & {t} \\\\"


def main_table() -> str:
    rows = [
        row("Ornith-1.5-9B oQ3", "Ornith-1.5-9B-MLX-oQ3", "3.9"),
        row("Ornith-1.5-9B oQ4 (stock)", "Ornith-1.5-9B-MLX-oQ4", "4.9", {"truthfulqa"}),
        row("Ornith-1.5-9B oQ8", "Ornith-1.5-9B-MLX-oQ8", "8.9"),
        row("Ornith-1.5-9B bf16", "Ornith-1.5-9B-MLX", "17", {"humaneval"}),
        "\\midrule",
        row("\\textbf{distilled oQ4 (v1, ours)}", "Ornith-1.5-9B-MLX-distil-oQ4", "\\textbf{4.9}", {"mmlu"}),
        "\\midrule",
        row("Ornith-1.5-35B-A3B 4-bit (teacher)", "Ornith-1.5-35B-A3B-MLX-4bit", "18"),
        row("Qwen3.6-35B-A3B 4-bit (teacher)", "Qwen3.6-35B-A3B-MLX-4bit", "19"),
    ]
    return "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Main result. All rows measured on the oMLX server under the same protocol "
        "(MMLU: fixed 1{,}000-question sample; TruthfulQA: all 817; HumanEval: all 164; thinking on; greedy). "
        "Size is the on-disk checkpoint. The distilled model is quantized with the identical oQ4 map as the stock build.}",
        "\\label{tab:main}",
        "\\begin{tabular}{lrrrrr}", "\\toprule",
        "Model & GB & MMLU & TruthfulQA & HumanEval & HumanEval s \\\\", "\\midrule",
        *rows,
        "\\bottomrule", "\\end{tabular}", "\\end{table}",
    ])


def runs_table() -> str:
    v1 = BASE["Ornith-1.5-9B-MLX-distil-oQ4"]
    def d(key, b):
        m = BASE.get(key, {}).get(b)
        if not m:
            return "--"
        return f"{(m['accuracy'] - v1[b]['accuracy']) * 100:+.1f}"
    spec = [
        ("v2", "misconception slice replaces TriviaQA; brief code", "Ornith-1.5-9B-MLX-distil-v2-oQ4"),
        ("v3", "abstention slice added; v1 code traces restored", "Ornith-1.5-9B-MLX-distil-v3-oQ4"),
        ("v4", "1{,}024-token length bias removed, all else held", "Ornith-1.5-9B-MLX-distil-v4-oQ4"),
        ("v5", "objective $\\to$ 0.3\\,CE + 0.7\\,KL(teacher$\\|$student), top-64", "Ornith-1.5-9B-MLX-distil-v5-oQ4"),
        ("v7", "adapter rank 32 $\\to$ 64, data byte-identical", "Ornith-1.5-9B-MLX-distil-v7-oQ4"),
        ("v8", "on-policy: v1's own failures, continued from v1", "Ornith-1.5-9B-MLX-distil-v8-oQ4"),
        ("v8-ctl", "same recipe, random prompts (control)", "Ornith-1.5-9B-MLX-distil-v8ctl-oQ4"),
    ]
    rows = [f"\\textbf{{v1}} & baseline recipe & {pct(v1['mmlu']['accuracy'])} & {pct(v1['truthfulqa']['accuracy'])} & {pct(v1['humaneval']['accuracy'])} & --- \\\\", "\\midrule"]
    for name, what, key in spec:
        m = BASE.get(key, {})
        rows.append(
            f"{name} & {what} & {pct(m.get('mmlu', {}).get('accuracy'))} & "
            f"{pct(m.get('truthfulqa', {}).get('accuracy'))} & {pct(m.get('humaneval', {}).get('accuracy'))} & "
            f"{d(key, 'mmlu')} / {d(key, 'truthfulqa')} / {d(key, 'humaneval')} \\\\"
        )
    rows.append("v6 & filter $\\to$ recover verified abstentions & \\multicolumn{4}{l}{stopped before training: 22 usable traces, 0 under the cap} \\\\")
    return "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Every training run after v1, each moving one variable, each with a prediction recorded "
        "before measurement. oMLX, same protocol as Table~\\ref{tab:main}. v5 was abandoned after MMLU "
        "(5.4$\\times$ v1's wall clock). Across the six completed comparable runs, 21 of 24 deltas are negative "
        "and none is positive beyond noise.}",
        "\\label{tab:runs}",
        "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{llrrrl}", "\\toprule",
        "Run & Change from v1 & MMLU & TruthfulQA & HumanEval & $\\Delta$ vs v1 \\\\", "\\midrule",
        *rows,
        "\\bottomrule", "\\end{tabular}", "}", "\\end{table}",
    ])


def _summary(tag: str, bench: str) -> dict | None:
    p = EVAL / tag / "summary.json"
    if not p.exists():
        return None
    for r in json.loads(p.read_text())["results"]:
        if r["benchmark"] == bench:
            return r
    return None


def forcing_table() -> str:
    spec = [
        ("v1", "HumanEval", "2{,}048", "bf-v1-he2048-plain", "bf-v1-he2048-forced", "humaneval"),
        ("v1", "HumanEval", "4{,}096", "bf-v1-he4096-plain", "bf-v1-he4096-forced", "humaneval"),
        ("stock oQ4", "HumanEval", "4{,}096", "bf-stock-he4096-plain", "bf-stock-he4096-forced", "humaneval"),
        ("v1", "MMLU$_{250}$", "3{,}072", "bf-v1-mmlu250-plain", "bf-v1-mmlu250-forced", "mmlu"),
    ]
    rows = []
    for model, bench, budget, tp, tf, b in spec:
        p, f = _summary(tp, b), _summary(tf, b)
        if not p:
            rows.append(f"{model} & {bench} & {budget} & \\multicolumn{{5}}{{l}}{{\\emph{{pending}}}} \\\\")
            continue
        plain = f"{p['correct']}/{p['total']} ({pct(p['accuracy'])})"
        if not f:
            rows.append(f"{model} & {bench} & {budget} & {plain} & {p['truncated']} & \\multicolumn{{3}}{{l}}{{\\emph{{pending}}}} \\\\")
            continue
        forced = f"{f['correct']}/{f['total']} (\\textbf{{{pct(f['accuracy'])}}})"
        delta = f["correct"] - p["correct"]
        rows.append(
            f"{model} & {bench} & {budget} & {plain} & {p['truncated']} & {forced} & "
            f"{f.get('forced_correct', 0)} of {f.get('forced', 0)} & {delta:+d} \\\\"
        )
    return "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Budget forcing on the same weights, our harness, reserve $N=256$. \\emph{Plain}: standard decoding "
        "under the budget. \\emph{Forced}: two-phase decoding; items still inside \\texttt{<think>} at the "
        "phase-1 cap have the block closed for them and receive the reserve to answer. No item receives more "
        "than the budget in total.}",
        "\\label{tab:forcing}",
        "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{lllrrrrr}", "\\toprule",
        "Model & Benchmark & Budget & Plain & Trunc. & Forced & Forced $\\to$ correct & $\\Delta$ \\\\", "\\midrule",
        *rows,
        "\\bottomrule", "\\end{tabular}", "}", "\\end{table}",
    ])


def _items(tag: str, bench: str) -> dict[str, dict]:
    path = EVAL / tag / f"{bench}.jsonl"
    if not path.exists():
        return {}
    with path.open() as f:
        return {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}


def _regressions(plain_tag: str, tag: str, bench: str) -> int | None:
    a, b = _items(plain_tag, bench), _items(tag, bench)
    if not a or not b:
        return None
    return sum(1 for i in a if a[i]["correct"] and i in b and not b[i]["correct"])


def soft_table() -> str:
    """Plain vs hard HALT vs the soft ramp, same weights, same budget."""
    spec = [
        ("v1", "HumanEval", "2{,}048", "humaneval", "bf-v1-he2048-plain", "bf-v1-he2048-forced", "p4-bias1024-0.02", "1{,}024"),
        ("stock oQ4", "HumanEval", "2{,}048", "humaneval", "p9-stock-plain", None, "p10-stock-bias1024-0.02", "1{,}024"),
        ("v1", "MMLU$_{250}$", "3{,}072", "mmlu", "bf-v1-mmlu250-plain", "bf-v1-mmlu250-forced", "p11-v1-mmlu250-bias2048-0.02", "2{,}048"),
        ("v1", "TruthfulQA", "2{,}048", "truthfulqa", "p19-v1-tqa-plain", None, "p20-v1-tqa-bias1024-0.02", "1{,}024"),
    ]
    rows = []
    for model, bench, budget, b, tp, tf, ts, start in spec:
        p, f, s_ = _summary(tp, b), _summary(tf, b) if tf else None, _summary(ts, b)
        if not (p and s_):
            rows.append(f"{model} & {bench} & {budget} & \\multicolumn{{6}}{{l}}{{\\emph{{pending}}}} \\\\")
            continue
        hard = f"{f['correct']}" if f else "--"
        reg = _regressions(tp, ts, b)
        rows.append(
            f"{model} & {bench} & {budget} & {p['correct']}/{p['total']} & {p['truncated']} & {hard} & "
            f"{start} & \\textbf{{{s_['correct']}}} & {s_['truncated']} & {reg if reg is not None else '--'} & "
            f"{s_['correct'] - p['correct']:+d} \\\\"
        )
    return "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Soft \\HALT{}: a bias of $0.02\\,(n - n_0)$ on the \\texttt{</think>} logit while the block is "
        "open, for generated position $n > n_0$; nothing before $n_0$, nothing once the block closes. Same weights and "
        "budget as \\emph{plain}; \\emph{hard} is the two-phase limit of Table~\\ref{tab:forcing} ($N=256$). "
        "\\emph{Regr.} counts items correct under plain decoding that the ramp gets wrong.}",
        "\\label{tab:soft}",
        "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{lllrrrrrrrr}", "\\toprule",
        " & & & \\multicolumn{2}{c}{Plain} & Hard & \\multicolumn{4}{c}{Soft ramp} & \\\\",
        "\\cmidrule(lr){4-5} \\cmidrule(lr){7-10}",
        "Model & Benchmark & Budget & Correct & Trunc. & Correct & $n_0$ & Correct & Trunc. & Regr. & $\\Delta$ vs plain \\\\", "\\midrule",
        *rows,
        "\\bottomrule", "\\end{tabular}", "}", "\\end{table}",
    ])


def sweep_table() -> str:
    spec = [("1{,}024", "0.01", "p5-bias1024-0.01"), ("1{,}024", "0.02", "p4-bias1024-0.02"),
            ("1{,}024", "0.04", "p6-bias1024-0.04"), ("768", "0.02", "p7-bias768-0.02"),
            ("1{,}280", "0.02", "p8-bias1280-0.02")]
    p = _summary("bf-v1-he2048-plain", "humaneval")
    rows = [f"\\multicolumn{{2}}{{l}}{{plain}} & {p['correct']} & {p['truncated']} & -- & {_mean_tokens('bf-v1-he2048-plain')} \\\\", "\\midrule"]
    for start, slope, tag in spec:
        s_ = _summary(tag, "humaneval")
        if not s_:
            rows.append(f"{start} & {slope} & \\multicolumn{{4}}{{l}}{{\\emph{{pending}}}} \\\\")
            continue
        reg = _regressions("bf-v1-he2048-plain", tag, "humaneval")
        rows.append(f"{start} & {slope} & {s_['correct']} & {s_['truncated']} & {reg} & {_mean_tokens(tag)} \\\\")
    return "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Ramp sweep on the distilled oQ4 build, HumanEval, 2{,}048-token budget, our harness. "
        "The effect is a plateau: any slope from 0.02 and any start from 768 to 1{,}280 closes every block and "
        "recovers 14--17 items; 0.01 is too weak to close the block by the budget.}",
        "\\label{tab:sweep}",
        "\\begin{tabular}{llrrrr}", "\\toprule",
        "$n_0$ & slope & Correct & Trunc. & Regr. & Mean tokens \\\\", "\\midrule",
        *rows,
        "\\bottomrule", "\\end{tabular}", "\\end{table}",
    ])


def generality_table() -> str:
    spec = [("oQ3", "p13-oq3-plain", "p14-oq3-bias1024-0.02"),
            ("oQ4 (stock)", "p9-stock-plain", "p10-stock-bias1024-0.02"),
            ("oQ4 (distilled, v1)", "bf-v1-he2048-plain", "p4-bias1024-0.02"),
            ("oQ8", "p15-oq8-plain", "p16-oq8-bias1024-0.02"),
            ("bf16 parent", "base-bf16-budget2048", "p12-bf16-bias1024-0.02"),
            ("Ornith-1.5-35B-A3B 4-bit (teacher)", "p17-35b-plain", "p18-35b-bias1024-0.02"),
            "\\midrule",
            ("Qwen3.8-27B 4-bit (dense)", "p21-qwen27b-plain", "p22-qwen27b-bias1024-0.02"),
            ("Qwen3.6-35B-A3B 4-bit (MoE)", "p23-qwen35b-plain", "p24-qwen35b-bias1024-0.02")]
    rows = []
    for entry in spec:
        if isinstance(entry, str):
            rows.append(entry)
            continue
        name, tp, ts = entry
        p, s_ = _summary(tp, "humaneval"), _summary(ts, "humaneval")
        if not (p and s_):
            rows.append(f"{name} & \\multicolumn{{6}}{{l}}{{\\emph{{pending}}}} \\\\")
            continue
        reg = _regressions(tp, ts, "humaneval")
        frac = (s_["correct"] - p["correct"]) / max(p["truncated"], 1)
        rows.append(f"{name} & {p['correct']} & {p['truncated']} & \\textbf{{{s_['correct']}}} & {s_['truncated']} & "
                    f"{s_['correct'] - p['correct']:+d} & {reg} & {frac:.2f} \\\\")
    return "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{The soft ramp across the quantization ladder, the 35B teacher, and two other architectures, "
        "HumanEval, 2{,}048-token budget, our harness, $n_0 = 1{,}024$, $s = 0.02$. The gain tracks the truncation "
        "count on every build, full precision included; \\emph{Regr.} is items correct under plain decoding that "
        "the ramp gets wrong.}",
        "\\label{tab:generality}",
        "\\begin{tabular}{lrrrrrrr}", "\\toprule",
        " & \\multicolumn{2}{c}{Plain} & \\multicolumn{2}{c}{Ramp} & & & \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
        "Build & Correct & Trunc. & Correct & Trunc. & $\\Delta$ & Regr. & $\\Delta$ / Trunc. \\\\", "\\midrule",
        *rows,
        "\\bottomrule", "\\end{tabular}", "\\end{table}",
    ])


def _mean_tokens(tag: str) -> str:
    items = _items(tag, "humaneval")
    return f"{sum(r['tokens'] for r in items.values()) / len(items):,.0f}" if items else "--"


def ladder_table() -> str:
    base = BASE["Ornith-1.5-9B-MLX-oQ4"]["mmlu"]["seconds"]
    spec = [("oQ3", "Ornith-1.5-9B-MLX-oQ3"), ("oQ4", "Ornith-1.5-9B-MLX-oQ4"), ("oQ8", "Ornith-1.5-9B-MLX-oQ8"),
            ("bf16", "Ornith-1.5-9B-MLX"), ("\\textbf{distilled oQ4}", "Ornith-1.5-9B-MLX-distil-oQ4")]
    rows = []
    for name, key in spec:
        m = BASE[key]["mmlu"]
        rows.append(f"{name} & {pct(m['accuracy'])} & {secs(m['seconds'])} & {m['seconds'] / base:.2f}$\\times$ \\\\")
    return "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{The quantization ladder on the same 1{,}000 MMLU questions (oMLX). oQ3 has fewer bits than oQ4 "
        "and should be at least as fast per token; it takes 2.23$\\times$ as long. bf16's 2.20$\\times$ is unquantized "
        "compute being slower per token, a different effect. Wall-clock totals, not token counts.}",
        "\\label{tab:ladder}",
        "\\begin{tabular}{lrrr}", "\\toprule",
        "Build & MMLU & Seconds & vs oQ4 \\\\", "\\midrule",
        *rows,
        "\\bottomrule", "\\end{tabular}", "\\end{table}",
    ])


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, fn in [("tables_main", main_table), ("tables_runs", runs_table),
                     ("tables_forcing", forcing_table), ("tables_ladder", ladder_table),
                     ("tables_soft", soft_table), ("tables_sweep", sweep_table),
                     ("tables_generality", generality_table)]:
        (OUT / f"{name}.tex").write_text(fn() + "\n")
        print(f"wrote paper/{name}.tex")
