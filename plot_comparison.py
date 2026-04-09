"""
Plot cross-perturbation-type comparisons from the averaged CSVs.
Uses only the 'proactive' condition (not 'proactive_original').

Generates 5 plots saved to <averaged_dir>/plots/:
  1. heatmap_pass1.png         — task_id x pert_type heatmap of avg_pass@1
  2. boxplot_all_metrics.png   — distribution per pert_type for all 3 pass@k metrics
  3. bar_mean_metrics.png      — grouped bar: mean pass@1/5/10 per pert_type
  4. category_box_pass1.png    — box distribution grouped by category (A/C/D/E/P/S)
  5. task_strip_pass1.png      — per-task dot plot across pert types (tasks shared by 3+ types)

Usage:
    python plot_comparison.py
    python plot_comparison.py --averaged_dir results/mbpp_codellama_proactive/averaged
    python plot_comparison.py --averaged_dir results/mbpp_codellama_proactive/averaged --out_dir plots/
"""

import argparse
import csv
import glob
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

PERT_ORDER = [
    "P1", "P2",
    "A1", "A2", "A3",
    "C1", "C2", "C3",
    "D1", "D2", "D3", "D4",
    "E1", "E2", "E3", "E4", "E5", "E6",
    "S1", "S2",
]

CATEGORY_MAP = {
    "P": ("Problem",    "#8172B2"),
    "A": ("Argument",   "#4C72B0"),
    "C": ("Control",    "#DD8452"),
    "D": ("Data",       "#55A868"),
    "E": ("Expression", "#C44E52"),
    "S": ("Statement",  "#937860"),
}

METRICS = ["avg_pass@1", "avg_pass@5", "avg_pass@10"]
METRIC_LABELS = ["pass@1", "pass@5", "pass@10"]
METRIC_COLORS = ["#C44E52", "#4C72B0", "#55A868"]


def cat_color(pert_type: str) -> str:
    return CATEGORY_MAP.get(pert_type[0].upper(), ("Other", "#888888"))[1]


def cat_name(pert_type: str) -> str:
    return CATEGORY_MAP.get(pert_type[0].upper(), ("Other", "#888888"))[0]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_averaged_dir(averaged_dir: str) -> dict:
    """
    Returns:
        {
          pert_type: {
            task_id (str): {
              "avg_pass@1": float,
              "avg_pass@5": float,
              "avg_pass@10": float,
              "avg_pass_ratio": float,
            }
          }
        }
    Only 'proactive' condition rows are kept.
    """
    data = {}
    pattern = os.path.join(averaged_dir, "*_avg.csv")
    for csv_path in sorted(glob.glob(pattern)):
        pert_type = os.path.basename(csv_path).replace("_avg.csv", "")
        task_data = {}
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("condition", "").strip() != "proactive":
                    continue
                tid = str(row["task_id"]).strip()
                entry = {}
                for m in ["avg_pass@1", "avg_pass@5", "avg_pass@10", "avg_pass_ratio"]:
                    try:
                        entry[m] = float(row[m])
                    except (ValueError, KeyError):
                        entry[m] = float("nan")
                task_data[tid] = entry
        if task_data:
            data[pert_type] = task_data
    return data


# ── Plot 1: Heatmap task_id × pert_type ──────────────────────────────────────

def plot_heatmap(data: dict, save_path: str):
    # Collect all task_ids and pert_types that are present
    pert_types = [p for p in PERT_ORDER if p in data]
    all_tasks  = sorted(
        {tid for pt in pert_types for tid in data[pt]},
        key=lambda x: int(x) if x.isdigit() else x,
    )

    # Build matrix — NaN where task not in that pert_type
    matrix = np.full((len(all_tasks), len(pert_types)), np.nan)
    for j, pt in enumerate(pert_types):
        for i, tid in enumerate(all_tasks):
            val = data[pt].get(tid, {}).get("avg_pass@1", np.nan)
            matrix[i, j] = val

    fig_h = max(8, len(all_tasks) * 0.28)
    fig_w = max(8, len(pert_types) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#DDDDDD")  # grey for missing

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")

    # Annotate cells
    for i in range(len(all_tasks)):
        for j in range(len(pert_types)):
            val = matrix[i, j]
            if not np.isnan(val):
                txt_color = "black" if 0.3 < val < 0.75 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.5, color=txt_color, fontweight="bold")

    # Axes
    ax.set_xticks(range(len(pert_types)))
    ax.set_xticklabels(pert_types, fontsize=9)
    ax.set_yticks(range(len(all_tasks)))
    ax.set_yticklabels(all_tasks, fontsize=7)
    ax.set_xlabel("Perturbation Type", fontsize=10)
    ax.set_ylabel("Task ID", fontsize=10)
    ax.set_title("avg_pass@1 per task per perturbation type  (proactive only)\nGrey = task not in that perturbation type", fontsize=11)

    # Colour the x-tick labels by category
    for tick, pt in zip(ax.get_xticklabels(), pert_types):
        tick.set_color(cat_color(pt))
        tick.set_fontweight("bold")

    # Category legend
    patches = [
        mpatches.Patch(color=c, label=name)
        for name, c in [v for v in CATEGORY_MAP.values()]
        if any(cat_name(pt) == name for pt in pert_types)
    ]
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.01, 1),
              fontsize=8, title="Category", title_fontsize=9)

    plt.colorbar(im, ax=ax, shrink=0.6, label="avg_pass@1")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ── Plot 2: Boxplot distribution of pass@k per pert_type ─────────────────────

def plot_boxplot_metrics(data: dict, save_path: str):
    pert_types = [p for p in PERT_ORDER if p in data]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, metric, label, color in zip(axes, METRICS, METRIC_LABELS, METRIC_COLORS):
        values_per_pt = []
        for pt in pert_types:
            vals = [v[metric] for v in data[pt].values()
                    if not np.isnan(v.get(metric, np.nan))]
            values_per_pt.append(vals)

        bp = ax.boxplot(
            values_per_pt,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=1),
            capprops=dict(linewidth=1),
            flierprops=dict(marker="o", markersize=3, alpha=0.5),
            widths=0.55,
        )
        for patch, pt in zip(bp["boxes"], pert_types):
            patch.set_facecolor(cat_color(pt))
            patch.set_alpha(0.75)

        ax.set_xticks(range(1, len(pert_types) + 1))
        ax.set_xticklabels(pert_types, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(label, fontsize=10)
        ax.set_ylim(-0.05, 1.1)
        ax.set_title(f"Distribution of {label}", fontsize=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        for tick, pt in zip(ax.get_xticklabels(), pert_types):
            tick.set_color(cat_color(pt))
            tick.set_fontweight("bold")

    # Shared category legend on the last axis
    patches = [
        mpatches.Patch(color=c, alpha=0.75, label=name)
        for name, c in [v for v in CATEGORY_MAP.values()]
        if any(cat_name(pt) == name for pt in pert_types)
    ]
    axes[-1].legend(handles=patches, fontsize=8, loc="upper right")

    fig.suptitle("Pass@k score distributions per perturbation type  (proactive, averaged per task)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ── Plot 3: Grouped bar — mean pass@1/5/10 per pert_type ─────────────────────

def plot_bar_mean_metrics(data: dict, save_path: str):
    pert_types = [p for p in PERT_ORDER if p in data]
    n_pts   = len(pert_types)
    n_bars  = 3
    width   = 0.25
    x       = np.arange(n_pts)
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(max(12, n_pts * 0.8), 6))

    for offset, metric, label, color in zip(offsets, METRICS, METRIC_LABELS, METRIC_COLORS):
        means = []
        for pt in pert_types:
            vals = [v[metric] for v in data[pt].values()
                    if not np.isnan(v.get(metric, np.nan))]
            means.append(np.mean(vals) if vals else np.nan)

        bars = ax.bar(x + offset, means, width=width, label=label,
                      color=color, alpha=0.82, edgecolor="white", linewidth=0.5)

        # Value label on top of each bar
        for bar, val in zip(bars, means):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                        f"{val:.2f}", ha="center", va="bottom",
                        fontsize=6, rotation=90, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(pert_types, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean score", fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_title("Mean pass@1 / pass@5 / pass@10 per perturbation type  (proactive only)", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    for tick, pt in zip(ax.get_xticklabels(), pert_types):
        tick.set_color(cat_color(pt))
        tick.set_fontweight("bold")

    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ── Plot 4: Category-level box distributions ──────────────────────────────────

def plot_category_box(data: dict, save_path: str):
    """One box per category, pooling all tasks × pert_types within it."""
    pert_types = [p for p in PERT_ORDER if p in data]

    # Group pass@1 values by category
    cat_vals = defaultdict(list)
    for pt in pert_types:
        cat = cat_name(pt)
        for v in data[pt].values():
            val = v.get("avg_pass@1", np.nan)
            if not np.isnan(val):
                cat_vals[cat].append(val)

    categories = [name for name, _ in CATEGORY_MAP.values() if name in cat_vals]
    colors      = [c   for name, c in CATEGORY_MAP.values() if name in cat_vals]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(
        [cat_vals[c] for c in categories],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
        widths=0.5,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)

    # Overlay individual points (jittered)
    for i, (cat, color) in enumerate(zip(categories, colors), start=1):
        y = cat_vals[cat]
        x_jitter = np.random.default_rng(42).uniform(-0.18, 0.18, len(y))
        ax.scatter(np.full(len(y), i) + x_jitter, y,
                   color=color, alpha=0.45, s=18, zorder=3)

    # Annotate n and mean
    for i, cat in enumerate(categories, start=1):
        vals = cat_vals[cat]
        ax.text(i, 1.05, f"n={len(vals)}\n{np.mean(vals):.3f}",
                ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(range(1, len(categories) + 1))
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(-0.05, 1.18)
    ax.set_ylabel("avg_pass@1", fontsize=10)
    ax.set_title("pass@1 distribution by perturbation category  (proactive, all tasks pooled)", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    for tick, color in zip(ax.get_xticklabels(), colors):
        tick.set_color(color)
        tick.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ── Plot 5: Per-task strip — tasks present in 3+ pert types ──────────────────

def plot_task_strip(data: dict, save_path: str):
    """
    For each task that appears in >= 3 perturbation types,
    plot its pass@1 score as a dot per pert_type, one row per task.
    Tasks are sorted by their mean pass@1 across all pert types they appear in.
    """
    pert_types = [p for p in PERT_ORDER if p in data]

    # Find tasks in >= 3 pert types
    task_pts = defaultdict(list)
    for pt in pert_types:
        for tid in data[pt]:
            task_pts[tid].append(pt)
    eligible = {tid: pts for tid, pts in task_pts.items() if len(pts) >= 3}

    if not eligible:
        print("  (no tasks appear in 3+ pert types — skipping strip plot)")
        return

    # Sort tasks by mean pass@1
    def task_mean(tid):
        vals = [data[pt][tid]["avg_pass@1"] for pt in eligible[tid]
                if not np.isnan(data[pt][tid].get("avg_pass@1", np.nan))]
        return np.mean(vals) if vals else 0.0

    sorted_tasks = sorted(eligible, key=task_mean)

    n_tasks = len(sorted_tasks)
    fig_h   = max(6, n_tasks * 0.32)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    y_pos = {tid: i for i, tid in enumerate(sorted_tasks)}

    for pt in pert_types:
        color = cat_color(pt)
        for tid, task_data in data[pt].items():
            if tid not in y_pos:
                continue
            val = task_data.get("avg_pass@1", np.nan)
            if np.isnan(val):
                continue
            ax.scatter(val, y_pos[tid], color=color, s=28, alpha=0.75,
                       linewidths=0, zorder=3)

    # Connect dots for each task with a thin line
    for tid in sorted_tasks:
        pts_for_task = [(data[pt][tid]["avg_pass@1"], y_pos[tid])
                        for pt in pert_types if tid in data[pt]
                        and not np.isnan(data[pt][tid].get("avg_pass@1", np.nan))]
        if len(pts_for_task) > 1:
            xs = [p[0] for p in pts_for_task]
            ax.plot([min(xs), max(xs)], [y_pos[tid], y_pos[tid]],
                    color="#AAAAAA", linewidth=0.6, zorder=1)

    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(list(y_pos.keys()), fontsize=7)
    ax.set_xlabel("avg_pass@1", fontsize=10)
    ax.set_xlim(-0.05, 1.1)
    ax.set_title(f"Per-task pass@1 across perturbation types  (tasks in 3+ pert types, n={n_tasks})\n"
                 "Each dot = one pert type; bar = range across types; tasks sorted by mean score",
                 fontsize=10)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Legend
    patches = [
        mpatches.Patch(color=c, label=name, alpha=0.8)
        for name, c in [v for v in CATEGORY_MAP.values()]
        if any(cat_name(pt) == name for pt in pert_types)
    ]
    ax.legend(handles=patches, fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot cross-perturbation-type comparisons from averaged CSVs."
    )
    parser.add_argument(
        "--averaged_dir",
        type=str,
        default="results/mbpp_codellama_proactive/averaged",
        help="Directory containing *_avg.csv files (default: results/mbpp_codellama_proactive/averaged).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for plots. Defaults to <averaged_dir>/plots/.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.averaged_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nLoading averaged CSVs from: {args.averaged_dir}")
    data = load_averaged_dir(args.averaged_dir)

    if not data:
        print("No data loaded. Check --averaged_dir.")
        return

    found_pts = [p for p in PERT_ORDER if p in data]
    print(f"Perturbation types found: {found_pts}\n")

    plot_heatmap(
        data,
        os.path.join(out_dir, "heatmap_pass1.png"),
    )
    plot_boxplot_metrics(
        data,
        os.path.join(out_dir, "boxplot_all_metrics.png"),
    )
    plot_bar_mean_metrics(
        data,
        os.path.join(out_dir, "bar_mean_metrics.png"),
    )
    plot_category_box(
        data,
        os.path.join(out_dir, "category_box_pass1.png"),
    )
    plot_task_strip(
        data,
        os.path.join(out_dir, "task_strip_pass1.png"),
    )

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
