"""Generate the paper's figures from measured artifacts.

    uv run --with matplotlib python scripts/make_paper_figures.py

Every point is read from `benchmarks/baseline.json` (oMLX) or from
`runs/v1/eval/<tag>/summary.json` (our harness), never typed in. Same role
as budgetbench/scripts/plot_tradeoffs.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper"
BASE = json.loads((ROOT / "benchmarks" / "baseline.json").read_text())["models"]
EVAL = ROOT / "runs" / "v1" / "eval"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})
C = {"stock": "#9a9a9a", "v1": "#1f5f8b", "bf16": "#2a2a2a", "forced": "#c0392b"}


def summary(tag: str, bench: str = "humaneval") -> dict | None:
    p = EVAL / tag / "summary.json"
    if not p.exists():
        return None
    for r in json.loads(p.read_text())["results"]:
        if r["benchmark"] == bench:
            return r
    return None


def fig_truncation() -> None:
    """Thesis in one plot: HumanEval accuracy against truncation rate, every
    (model, budget, decoder) point our harness has measured on all 164 items."""
    pts = [
        ("stock oQ4, 4096", "bf-stock-he4096-plain", C["stock"], "o"),
        ("stock oQ4, 4096 + HALT", "bf-stock-he4096-forced", C["stock"], "*"),
        ("distilled, 2048", "bf-v1-he2048-plain", C["v1"], "o"),
        ("distilled, 2048 + HALT", "bf-v1-he2048-forced", C["v1"], "*"),
        ("distilled, 4096", "bf-v1-he4096-plain", C["v1"], "s"),
        ("distilled, 4096 + HALT", "bf-v1-he4096-forced", C["v1"], "*"),
        ("bf16, 2048", "base-bf16-budget2048", C["bf16"], "o"),
        ("bf16, 4096", "base-bf16", C["bf16"], "s"),
    ]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for label, tag, col, mk in pts:
        s = summary(tag)
        if not s:
            continue
        x = 100 * s["truncated"] / s["total"]
        y = 100 * s["accuracy"]
        ax.scatter(x, y, c=col, marker=mk, s=70 if mk == "*" else 36, zorder=3,
                   edgecolor="white", linewidth=0.5)
        dx, dy = (0.6, 0.4) if "HALT" not in label else (0.6, -1.2)
        ax.annotate(label, (x, y), (x + dx, y + dy), fontsize=6.5, color=col)
    ax.set_xlabel("problems truncated before answering (%)")
    ax.set_ylabel("HumanEval pass@1 (%)")
    ax.set_title("Accuracy tracks truncation, not bit width (our harness, n = 164)")
    ax.set_xlim(left=-1)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_truncation.pdf")


def fig_ladder() -> None:
    """Quantization ladder: MMLU accuracy against wall clock for the same
    1,000 questions. Fewer bits should mean faster; oQ3 is 2.2x slower."""
    spec = [("oQ3", "Ornith-1.5-9B-MLX-oQ3", C["stock"]), ("oQ4", "Ornith-1.5-9B-MLX-oQ4", C["stock"]),
            ("oQ8", "Ornith-1.5-9B-MLX-oQ8", C["stock"]), ("bf16", "Ornith-1.5-9B-MLX", C["bf16"]),
            ("distilled oQ4", "Ornith-1.5-9B-MLX-distil-oQ4", C["v1"])]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    offsets = {"oQ8": (0.15, -0.9), "distilled oQ4": (0.15, 0.5), "bf16": (-0.6, 0.5)}
    for name, key, col in spec:
        m = BASE[key]["mmlu"]
        x, y = m["seconds"] / 3600, 100 * m["accuracy"]
        ax.scatter(x, y, c=col, s=48, zorder=3, edgecolor="white", linewidth=0.5)
        dx, dy = offsets.get(name, (0.15, 0.4))
        ax.annotate(name, (x, y), (x + dx, y + dy), fontsize=7, color=col)
    ax.set_xlabel("wall clock for 1,000 MMLU questions (hours, oMLX)")
    ax.set_ylabel("MMLU accuracy (%)")
    ax.set_title("The 3-bit build is slower than the 4-bit build, and worse")
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ladder.pdf")


def fig_halt() -> None:
    """HALT effect: plain vs forced per (model, budget), correct counts."""
    spec = [("stock oQ4\n4,096", "bf-stock-he4096-plain", "bf-stock-he4096-forced"),
            ("distilled\n2,048", "bf-v1-he2048-plain", "bf-v1-he2048-forced"),
            ("distilled\n4,096", "bf-v1-he4096-plain", "bf-v1-he4096-forced")]
    labels, plain, forced, trunc = [], [], [], []
    for lab, tp, tf in spec:
        p, f = summary(tp), summary(tf)
        if not (p and f):
            continue
        labels.append(lab); plain.append(p["correct"]); forced.append(f["correct"]); trunc.append(p["truncated"])
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    xs = range(len(labels)); w = 0.36
    ax.bar([x - w / 2 for x in xs], plain, w, color=C["stock"], label="plain decoding")
    ax.bar([x + w / 2 for x in xs], forced, w, color=C["forced"], label="+ HALT (N = 256)")
    for x, p, f, t in zip(xs, plain, forced, trunc):
        ax.text(x - w / 2, p + 1.5, f"{p}\n({t} trunc.)", ha="center", fontsize=6.5)
        ax.text(x + w / 2, f + 1.5, f"{f}\n(+{f - p})", ha="center", fontsize=6.5, color=C["forced"])
    ax.axhline(145, color=C["bf16"], linewidth=0.8, linestyle="--")
    ax.text(-0.45, 146.5, "bf16 parent, 4,096", ha="left", fontsize=6.5, color=C["bf16"])
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels)
    ax.set_ylabel("HumanEval problems solved (of 164)")
    ax.set_ylim(100, 164)
    ax.set_title("HALT on the same weights: no training, no extra budget")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_halt.pdf")


def fig_runs() -> None:
    """Every completed run's delta against v1, per benchmark."""
    v1 = BASE["Ornith-1.5-9B-MLX-distil-oQ4"]
    runs = [("v2", "Ornith-1.5-9B-MLX-distil-v2-oQ4"), ("v3", "Ornith-1.5-9B-MLX-distil-v3-oQ4"),
            ("v4", "Ornith-1.5-9B-MLX-distil-v4-oQ4"), ("v5", "Ornith-1.5-9B-MLX-distil-v5-oQ4"),
            ("v7", "Ornith-1.5-9B-MLX-distil-v7-oQ4"), ("v8", "Ornith-1.5-9B-MLX-distil-v8-oQ4"),
            ("v8-ctl", "Ornith-1.5-9B-MLX-distil-v8ctl-oQ4")]
    benches = [("mmlu", "MMLU", "#1f5f8b"), ("truthfulqa", "TruthfulQA", "#7d3c98"), ("humaneval", "HumanEval", "#c0392b")]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for i, (name, key) in enumerate(runs):
        for j, (b, lab, col) in enumerate(benches):
            m = BASE.get(key, {}).get(b)
            if not m:
                continue
            d = 100 * (m["accuracy"] - v1[b]["accuracy"])
            ax.scatter(i + (j - 1) * 0.22, d, c=col, s=28, zorder=3, label=lab if i == 0 else None)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(runs))); ax.set_xticklabels([r[0] for r in runs])
    ax.set_ylabel("delta vs v1 (points)")
    ax.set_title("Every run after v1, against v1 (oMLX). v5: MMLU only.")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_runs.pdf")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for fn in (fig_truncation, fig_ladder, fig_halt, fig_runs):
        fn()
        print(f"wrote paper/{fn.__name__[4:]}: fig_{fn.__name__[4:]}.pdf")
