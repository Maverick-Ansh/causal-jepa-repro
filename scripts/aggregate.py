"""
Aggregate sweep results into the tables and figures used in REPORT.md.

    python -m scripts.aggregate --sweep results/sweep --out results --figs figures
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1-3 of the validated reference palette (light mode).
# Validated with scripts/validate_palette.js --mode light --pairs all:
#   worst CVD dE 9.2, worst normal-vision dE 24.0, all checks pass.
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdcd8"

ENC_LABEL = {"oracle": "oracle slots  (~VideoSAUR)",
             "degraded": "degraded slots  (~SAVi)"}
ENC_COLOR = {"oracle": C_BLUE, "degraded": C_ORANGE}
CATS = ["average", "descriptive", "predictive", "counterfactual", "explanatory"]


def style(ax, xlabel, ylabel, title=None):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)


def load(sweep):
    runs, controls = [], {}
    for f in sorted(glob.glob(os.path.join(sweep, "*.json"))):
        j = json.load(open(f))
        if os.path.basename(f).startswith("CONTROL"):
            controls[(j["encoder"], j["control"])] = j["vqa"]
        elif "n_mask" in j:
            runs.append(j)
    return runs, controls


def group(runs, getter):
    """(encoder, |M|) -> (mean, std, n) of getter(run), skipping missing values."""
    acc = defaultdict(list)
    for r in runs:
        v = getter(r)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        acc[(r["encoder"], r["n_mask"])].append(v)
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in acc.items()}


def series(g, enc, masks):
    m = [g.get((enc, k), (np.nan, 0, 0))[0] for k in masks]
    s = [g.get((enc, k), (np.nan, 0, 0))[1] for k in masks]
    return np.array(m), np.array(s)


def md_table(runs, controls, masks, encoders):
    """Tab. 1 reproduction: VQA accuracy vs |M|, with deltas against |M| = 0."""
    lines = []
    gs = {c: group(runs, lambda r, c=c: r.get("vqa", {}).get(c)) for c in CATS}
    lines.append("| encoder | model | \\|M\\| | " + " | ".join(
        c.capitalize() for c in CATS) + " |")
    lines.append("|---|---|---|" + "---|" * len(CATS))
    for enc in encoders:
        base = {c: gs[c].get((enc, 0), (np.nan,))[0] for c in CATS}
        for k in masks:
            cells = []
            for c in CATS:
                v = gs[c].get((enc, k))
                if v is None:
                    cells.append("—"); continue
                mu, sd, n = v
                cell = f"{mu:.2f}"
                if n > 1:
                    cell += f" ±{sd:.2f}"
                if k != 0 and not np.isnan(base[c]):
                    d = mu - base[c]
                    cell += f" ({'+' if d >= 0 else ''}{d:.2f})"
                cells.append(cell)
            name = "OC-JEPA" if k == 0 else "C-JEPA"
            lines.append(f"| {enc} | {name} | {k} | " + " | ".join(cells) + " |")
        for mode in ("oracle", "static"):
            cv = controls.get((enc, mode))
            if cv:
                lab = ("*ceiling* (true future)" if mode == "oracle"
                       else "*floor* (no dynamics)")
                lines.append(f"| {enc} | {lab} | – | " +
                             " | ".join(f"{cv[c]:.2f}" for c in CATS) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="/kaggle/working/results/sweep")
    ap.add_argument("--out", default="/kaggle/working/results")
    ap.add_argument("--figs", default="/kaggle/working/figures")
    args = ap.parse_args()
    os.makedirs(args.figs, exist_ok=True)

    runs, controls = load(args.sweep)
    if not runs:
        print("no runs found"); return
    masks = sorted({r["n_mask"] for r in runs})
    encoders = [e for e in ["oracle", "degraded"] if any(r["encoder"] == e for r in runs)]
    print(f"{len(runs)} runs, encoders={encoders}, |M|={masks}")

    # ---------------------------------------------------- Fig 1: VQA vs |M|
    panels = [("average", "VQA accuracy, all questions"),
              ("counterfactual", "Counterfactual questions only")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")
    for ax, (cat, title) in zip(axes, panels):
        g = group(runs, lambda r, c=cat: r.get("vqa", {}).get(c))
        for enc in encoders:
            mu, sd = series(g, enc, masks)
            ax.errorbar(masks, mu, yerr=sd, color=ENC_COLOR[enc], linewidth=2,
                        marker="o", markersize=7, capsize=3, label=ENC_LABEL[enc],
                        markeredgecolor="white", markeredgewidth=1.2)
            if not np.isnan(mu).all():
                i = int(np.nanargmax(mu))
                ax.annotate(f"{mu[i]:.1f}", (masks[i], mu[i]), textcoords="offset points",
                            xytext=(0, 10), ha="center", color=ENC_COLOR[enc], fontsize=9)
        for enc in encoders:
            cv = controls.get((enc, "static"))
            if cv:
                ax.axhline(cv[cat], color=ENC_COLOR[enc], linestyle=":", linewidth=1.4,
                           alpha=0.65)
        if cat == "counterfactual":
            ax.axhline(50.0, color=INK2, linestyle="--", linewidth=1.2, alpha=0.6)
            ax.annotate("copy-the-factual-outcome baseline (50%)", (masks[-1], 50.0),
                        textcoords="offset points", xytext=(-4, 6), ha="right",
                        color=INK2, fontsize=8)
        style(ax, "object masking budget  |M|   (0 = OC-JEPA)", "accuracy (%)", title)
        ax.set_xticks(masks)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle("Reproducing Tab. 1 — object-level masking vs VQA accuracy",
                 color=INK, fontsize=13, x=0.02, ha="left")
    fig.text(0.02, 0.005, "dotted = static (no-dynamics) floor for that encoder",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(os.path.join(args.figs, "fig1_vqa_vs_mask.png"), dpi=160)
    plt.close(fig)

    # ------------------------------------- Fig 2: influence neighborhoods
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")
    for ax, probe in zip(axes, ["attention", "ablation"]):
        g = group(runs, lambda r, p=probe: r.get("influence", {}).get(p, {}).get("auroc"))
        for enc in encoders:
            mu, sd = series(g, enc, masks)
            ax.errorbar(masks, mu, yerr=sd, color=ENC_COLOR[enc], linewidth=2,
                        marker="o", markersize=7, capsize=3, label=ENC_LABEL[enc],
                        markeredgecolor="white", markeredgewidth=1.2)
        ax.axhline(0.5, color=INK2, linestyle="--", linewidth=1.2, alpha=0.6)
        ax.annotate("chance", (masks[-1], 0.5), textcoords="offset points",
                    xytext=(-2, 5), ha="right", color=INK2, fontsize=8)
        title = ("Attention proxy (the paper's App. J measure)" if probe == "attention"
                 else "Causal ablation probe (ours)")
        style(ax, "object masking budget  |M|", "AUROC vs true interaction graph", title)
        ax.set_xticks(masks)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle("Do influence neighborhoods match the true interaction graph? (Cor. 1)",
                 color=INK, fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(args.figs, "fig2_influence_vs_mask.png"), dpi=160)
    plt.close(fig)

    # ------------------------------- Fig 3: probe-free prediction quality
    specs = [(lambda r: r.get("counterfactual", {}).get("cf_gain"),
              "counterfactual gain   (1 = perfect, 0 = ignored)",
              "Counterfactual rollout under do(remove k)"),
             (lambda r: r.get("collision", {}).get("collision_f1"),
              "collision F1 over imagined horizon",
              "Are the imagined interactions real?"),
             (lambda r: r["final"]["val_pos_err"],
              "mean position error (world is 1x1)",
              "Plain forward prediction")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor="white")
    for ax, (getter, ylab, title) in zip(axes, specs):
        g = group(runs, getter)
        for enc in encoders:
            mu, sd = series(g, enc, masks)
            ax.errorbar(masks, mu, yerr=sd, color=ENC_COLOR[enc], linewidth=2,
                        marker="o", markersize=7, capsize=3, label=ENC_LABEL[enc],
                        markeredgecolor="white", markeredgewidth=1.2)
        if "counterfactual gain" in ylab:
            ax.axhline(0.0, color=INK2, linestyle="--", linewidth=1.2, alpha=0.6)
            ax.annotate("ignores the intervention", (masks[-1], 0.0),
                        textcoords="offset points", xytext=(-2, 5), ha="right",
                        color=INK2, fontsize=8)
        style(ax, "object masking budget  |M|", ylab, title)
        ax.set_xticks(masks)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle("Probe-free measurements of the world model itself",
                 color=INK, fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(args.figs, "fig3_direct_metrics.png"), dpi=160)
    plt.close(fig)

    # ---------------------------------------------------------------- tables
    tbl = md_table(runs, controls, masks, encoders)
    summary = {
        "n_runs": len(runs), "masks": masks, "encoders": encoders,
        "vqa": {c: {f"{e}_M{k}": group(runs, lambda r, c=c: r.get("vqa", {}).get(c))
                    .get((e, k)) for e in encoders for k in masks} for c in CATS},
        "influence": {p: {f"{e}_M{k}": group(
            runs, lambda r, p=p: r.get("influence", {}).get(p, {}).get("auroc")).get((e, k))
            for e in encoders for k in masks} for p in ["attention", "ablation"]},
        "counterfactual": {f"{e}_M{k}": group(
            runs, lambda r: r.get("counterfactual", {}).get("cf_gain")).get((e, k))
            for e in encoders for k in masks},
        "collision_f1": {f"{e}_M{k}": group(
            runs, lambda r: r.get("collision", {}).get("collision_f1")).get((e, k))
            for e in encoders for k in masks},
        "controls": {f"{e}_{m}": v for (e, m), v in controls.items()},
    }
    os.makedirs(args.out, exist_ok=True)
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    open(os.path.join(args.out, "table1.md"), "w").write(tbl + "\n")
    print(tbl)
    print(f"\nwrote {args.out}/summary.json, {args.out}/table1.md, and 3 figures")


if __name__ == "__main__":
    main()
