"""
Analyze and plot evaluation results for a proactively fine-tuned model.

Usage:
    python analyze_results.py --results_dir results/mbpp_codellama_proactive
    python analyze_results.py --results_dir results/mbpp_codellama_proactive --metric pass@1
    python analyze_results.py --results_dir results/mbpp_codellama_proactive --save_dir plots/
"""

import argparse
import os
import glob
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")          # headless-safe; switch to "TkAgg" if you want interactive windows
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ──────────────────────────────────────────────
# Perturbation type groupings (prefix → category)
# ──────────────────────────────────────────────
CATEGORY_MAP = {
    "A": "Argument",
    "C": "Control",
    "D": "Data",
    "E": "Expression",
    "P": "Problem",
    "S": "Statement",
}

CATEGORY_COLORS = {
    "Argument":   "#4C72B0",
    "Control":    "#DD8452",
    "Data":       "#55A868",
    "Expression": "#C44E52",
    "Problem":    "#8172B2",
    "Statement":  "#937860",
}

# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_all_results(results_dir: str) -> list[dict]:
    """
    Walk results_dir/*/eval_results.csv and return every row as a dict.
    The condition column distinguishes:
      - perturbed  → the label written during eval (e.g. 'proactive', 'baseline')
      - original   → the same label + '_original'  (e.g. 'proactive_original')
    We normalise these to prompt_type = 'perturbed' | 'original'.
    """
    pattern = os.path.join(results_dir, "*", "eval_results.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        raise FileNotFoundError(
            f"No eval_results.csv files found under '{results_dir}'. "
            "Check that --results_dir points to the right folder."
        )

    rows = []
    for csv_path in csv_files:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalise numeric columns
                for col in ("pass@1", "pass@5", "pass@10", "pass_ratio"):
                    try:
                        row[col] = float(row[col])
                    except (ValueError, KeyError):
                        row[col] = float("nan")

                condition = row.get("condition", "")
                row["prompt_type"] = "original" if condition.endswith("_original") else "perturbed"
                rows.append(row)

    print(f"Loaded {len(rows)} rows from {len(csv_files)} CSV files.")
    return rows


# ──────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────

def aggregate(rows: list[dict], metric: str = "pass@1") -> dict:
    """
    Returns a dict:
      {
        pert_type: {
          "perturbed": mean_metric,
          "original":  mean_metric,
          "delta":     perturbed - original,
          "n":         number of tasks,
        }
      }
    """
    buckets = defaultdict(lambda: {"perturbed": [], "original": []})

    for row in rows:
        pt = row.get("pert_type", "").strip()
        if not pt:
            continue
        val = row.get(metric, float("nan"))
        if val != val:          # NaN check
            continue
        buckets[pt][row["prompt_type"]].append(val)

    result = {}
    for pt, data in sorted(buckets.items()):
        pert_vals = data["perturbed"]
        ori_vals  = data["original"]
        mean_pert = np.mean(pert_vals) if pert_vals else float("nan")
        mean_ori  = np.mean(ori_vals)  if ori_vals  else float("nan")
        result[pt] = {
            "perturbed": mean_pert,
            "original":  mean_ori,
            "delta":     mean_pert - mean_ori,
            "n":         max(len(pert_vals), len(ori_vals)),
        }
    return result


def category_of(pert_type: str) -> str:
    prefix = pert_type[0].upper()
    return CATEGORY_MAP.get(prefix, "Other")


# ──────────────────────────────────────────────
# Console summary
# ──────────────────────────────────────────────

def print_summary(agg: dict, metric: str):
    col_w = 12
    header = (
        f"{'pert_type':<{col_w}}  {'category':<12}  {'n':>5}  "
        f"{'perturbed':>10}  {'original':>10}  {'delta':>8}"
    )
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print(f"  Metric: {metric}")
    print(header)
    print(sep)

    grand_pert, grand_ori = [], []

    for pt, v in sorted(agg.items()):
        cat = category_of(pt)
        pert = v["perturbed"]
        ori  = v["original"]
        delta = v["delta"]
        delta_str = f"{delta:+.4f}" if delta == delta else "  N/A"
        print(
            f"{pt:<{col_w}}  {cat:<12}  {v['n']:>5}  "
            f"{pert:>10.4f}  {ori:>10.4f}  {delta_str:>8}"
        )
        if pert == pert:
            grand_pert.append(pert)
        if ori == ori:
            grand_ori.append(ori)

    print(sep)
    gp = np.mean(grand_pert) if grand_pert else float("nan")
    go = np.mean(grand_ori)  if grand_ori  else float("nan")
    gd = gp - go
    print(
        f"{'OVERALL':<{col_w}}  {'':12}  {'':>5}  "
        f"{gp:>10.4f}  {go:>10.4f}  {gd:>+8.4f}"
    )
    print("=" * len(header) + "\n")


# ──────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────

def _bar_positions(n_groups: int, n_bars: int, bar_width: float = 0.35, gap: float = 0.1):
    """Return x positions for each group × bar combination."""
    group_width = n_bars * bar_width + gap
    x = np.arange(n_groups) * group_width
    offsets = [(i - (n_bars - 1) / 2) * bar_width for i in range(n_bars)]
    return x, offsets, group_width


def plot_perturbed_vs_original(agg: dict, metric: str, save_path: str):
    """Grouped bar chart: perturbed vs original for every perturbation type."""
    pert_types = sorted(agg.keys())
    perturbed_vals = [agg[pt]["perturbed"] for pt in pert_types]
    original_vals  = [agg[pt]["original"]  for pt in pert_types]

    x, offsets, gw = _bar_positions(len(pert_types), 2)

    fig, ax = plt.subplots(figsize=(max(12, len(pert_types) * 0.9), 6))

    bars_pert = ax.bar(x + offsets[0], perturbed_vals, width=0.35,
                       label="Perturbed", color="#C44E52", alpha=0.85, edgecolor="white")
    bars_ori  = ax.bar(x + offsets[1], original_vals,  width=0.35,
                       label="Original",  color="#4C72B0", alpha=0.85, edgecolor="white")

    # Colour bars by category
    for bar, pt in zip(bars_pert, pert_types):
        bar.set_facecolor(CATEGORY_COLORS.get(category_of(pt), "#888"))
        bar.set_alpha(0.9)
    for bar in bars_ori:
        bar.set_facecolor("#AAAAAA")
        bar.set_alpha(0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(pert_types, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Mean {metric} — Perturbed vs Original prompt, by perturbation type")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Category legend patches
    cat_patches = [
        mpatches.Patch(color=c, label=cat)
        for cat, c in CATEGORY_COLORS.items()
        if any(category_of(pt) == cat for pt in pert_types)
    ]
    ori_patch = mpatches.Patch(color="#AAAAAA", alpha=0.7, label="Original (all types)")
    ax.legend(handles=cat_patches + [ori_patch], fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_delta(agg: dict, metric: str, save_path: str):
    """Horizontal bar chart showing delta = perturbed − original per pert_type."""
    pert_types = sorted(agg.keys())
    deltas = [agg[pt]["delta"] for pt in pert_types]

    colors = [
        CATEGORY_COLORS.get(category_of(pt), "#888")
        for pt in pert_types
    ]

    fig, ax = plt.subplots(figsize=(8, max(5, len(pert_types) * 0.45)))
    y = np.arange(len(pert_types))

    bars = ax.barh(y, deltas, color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(pert_types, fontsize=9)
    ax.set_xlabel(f"Δ {metric}  (perturbed − original)")
    ax.set_title(f"Performance delta on perturbed vs original prompts ({metric})")
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Value labels
    for bar, val in zip(bars, deltas):
        x_off = 0.005 if val >= 0 else -0.005
        ha = "left" if val >= 0 else "right"
        ax.text(val + x_off, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=7.5)

    cat_patches = [
        mpatches.Patch(color=c, label=cat)
        for cat, c in CATEGORY_COLORS.items()
        if any(category_of(pt) == cat for pt in pert_types)
    ]
    ax.legend(handles=cat_patches, fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_all_metrics(agg: dict, save_path: str):
    """
    4-panel figure: one panel per metric (pass@1, pass@5, pass@10, pass_ratio).
    Each panel shows perturbed (solid) and original (dashed) as a line over pert_types.
    """
    metrics = ["pass@1", "pass@5", "pass@10", "pass_ratio"]
    pert_types = sorted(agg.keys())
    x = np.arange(len(pert_types))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        # Recompute per-metric aggregation from the stored dicts
        # (we pass the full rows-based agg dict, so we need to re-aggregate per metric)
        pass  # handled below

    plt.close()

    # We need per-metric aggregations — callers pass full `rows`; handle below.
    # This function is called with already-aggregated data for ONE metric.
    # So we only have pass@1 data in `agg`.  Use a separate rows-based path.
    print("(plot_all_metrics requires rows; skipped in single-metric mode)")


def plot_all_metrics_from_rows(rows: list[dict], save_path: str):
    """Line plot: perturbed vs original across all 4 metrics in a 2×2 grid."""
    metrics = ["pass@1", "pass@5", "pass@10", "pass_ratio"]
    metric_labels = ["pass@1", "pass@5", "pass@10", "pass ratio"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    axes = axes.flatten()

    # Collect all pert_types across all aggregations
    all_pts = sorted({r.get("pert_type", "").strip() for r in rows if r.get("pert_type", "").strip()})

    for ax, metric, label in zip(axes, metrics, metric_labels):
        agg = aggregate(rows, metric)
        pert_types = sorted(agg.keys())
        x = np.arange(len(pert_types))

        perturbed_vals = [agg[pt]["perturbed"] for pt in pert_types]
        original_vals  = [agg[pt]["original"]  for pt in pert_types]

        # Color each point by category
        for i, pt in enumerate(pert_types):
            c = CATEGORY_COLORS.get(category_of(pt), "#888")
            ax.bar(i - 0.2, perturbed_vals[i], width=0.35, color=c, alpha=0.9, edgecolor="white")
            ax.bar(i + 0.2, original_vals[i],  width=0.35, color="#AAAAAA", alpha=0.7, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(pert_types, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(label, fontsize=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    # Shared legend
    cat_patches = [
        mpatches.Patch(color=c, label=cat, alpha=0.9)
        for cat, c in CATEGORY_COLORS.items()
        if any(category_of(pt) == cat for pt in all_pts)
    ]
    cat_patches.append(mpatches.Patch(color="#AAAAAA", alpha=0.7, label="Original"))
    fig.legend(handles=cat_patches, loc="lower center", ncol=len(cat_patches),
               fontsize=8, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle("Model performance: Perturbed vs Original prompts across all metrics", fontsize=13)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_category_summary(rows: list[dict], metric: str, save_path: str):
    """Grouped bar: average metric per perturbation category (perturbed vs original)."""
    agg = aggregate(rows, metric)

    cat_pert  = defaultdict(list)
    cat_ori   = defaultdict(list)
    for pt, v in agg.items():
        cat = category_of(pt)
        if v["perturbed"] == v["perturbed"]:
            cat_pert[cat].append(v["perturbed"])
        if v["original"] == v["original"]:
            cat_ori[cat].append(v["original"])

    categories = sorted(cat_pert.keys())
    mean_pert = [np.mean(cat_pert[c]) for c in categories]
    mean_ori  = [np.mean(cat_ori[c])  for c in categories]

    x, offsets, _ = _bar_positions(len(categories), 2, bar_width=0.3)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (cat, mp, mo) in enumerate(zip(categories, mean_pert, mean_ori)):
        color = CATEGORY_COLORS.get(cat, "#888")
        ax.bar(x[i] + offsets[0], mp, width=0.3, color=color, alpha=0.9,
               edgecolor="white", label=f"{cat} (pert)" if i == 0 else "_")
        ax.bar(x[i] + offsets[1], mo, width=0.3, color=color, alpha=0.45,
               edgecolor="white", hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel(f"Mean {metric}")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Mean {metric} by perturbation category — Perturbed (solid) vs Original (hatched)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    cat_patches = [
        mpatches.Patch(color=CATEGORY_COLORS.get(c, "#888"), label=c)
        for c in categories
    ]
    solid_patch  = mpatches.Patch(color="gray", alpha=0.9, label="Perturbed")
    hatch_patch  = mpatches.Patch(color="gray", alpha=0.45, hatch="//", label="Original")
    ax.legend(handles=cat_patches + [solid_patch, hatch_patch], fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze and plot proactive model evaluation results."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Root results directory, e.g. results/mbpp_codellama_proactive",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="pass@1",
        choices=["pass@1", "pass@5", "pass@10", "pass_ratio"],
        help="Primary metric to use for analysis (default: pass@1).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save plots. Defaults to <results_dir>/plots/.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Print the summary table only; skip generating plots.",
    )
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.join(args.results_dir, "plots")

    # ── Load ────────────────────────────────────
    rows = load_all_results(args.results_dir)

    # ── Console summary ─────────────────────────
    agg = aggregate(rows, args.metric)
    print_summary(agg, args.metric)

    if args.no_plots:
        return

    os.makedirs(save_dir, exist_ok=True)

    # ── Plot 1: perturbed vs original per pert_type ──
    plot_perturbed_vs_original(
        agg, args.metric,
        os.path.join(save_dir, f"perturbed_vs_original_{args.metric.replace('@', '')}.png"),
    )

    # ── Plot 2: delta per pert_type ──────────────
    plot_delta(
        agg, args.metric,
        os.path.join(save_dir, f"delta_{args.metric.replace('@', '')}.png"),
    )

    # ── Plot 3: all 4 metrics in a 2×2 grid ─────
    plot_all_metrics_from_rows(
        rows,
        os.path.join(save_dir, "all_metrics_overview.png"),
    )

    # ── Plot 4: category-level summary ──────────
    plot_category_summary(
        rows, args.metric,
        os.path.join(save_dir, f"category_summary_{args.metric.replace('@', '')}.png"),
    )

    print(f"\nAll plots saved to: {save_dir}")


if __name__ == "__main__":
    main()
