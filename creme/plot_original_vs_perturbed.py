import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


PERT_TYPES = [
    "A1", "A2", "A3",
    "C1", "C2", "C3",
    "D1", "D2", "D3", "D4",
    "E1", "E2", "E3", "E4", "E5", "E6",
    "P1", "P2",
    "S1", "S2",
]


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def load_scores(results_root: str, metric: str):
    grouped = defaultdict(lambda: {"perturbed": [], "original": []})

    for pert_type in PERT_TYPES:
        csv_path = os.path.join(results_root, pert_type, "eval_results.csv")
        if not os.path.exists(csv_path):
            continue

        with open(csv_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                bucket = "original" if "original" in row.get("condition", "").lower() else "perturbed"
                grouped[pert_type][bucket].append(float(row[metric]))

    summary = {}
    for pert_type, scores in grouped.items():
        summary[pert_type] = {
            "perturbed": _mean(scores["perturbed"]),
            "original": _mean(scores["original"]),
        }
    return summary


def plot_scores(summary, metric: str, title: str, output_path: str, perturbed_label: str, original_label: str):
    pert_types = [pt for pt in PERT_TYPES if pt in summary]
    perturbed_values = [summary[pt]["perturbed"] for pt in pert_types]
    original_values = [summary[pt]["original"] for pt in pert_types]

    x = list(range(len(pert_types)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar([i - width / 2 for i in x], perturbed_values, width=width, label=perturbed_label, color="#D95F02")
    ax.bar([i + width / 2 for i in x], original_values, width=width, label=original_label, color="#1B9E77")

    ax.set_title(title)
    ax.set_xlabel("Perturbation Type")
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(pert_types)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot averaged original vs perturbed scores by perturbation type from eval_results.csv files."
    )
    parser.add_argument(
        "--results_root",
        required=True,
        help="Run directory containing per-perturbation eval_results.csv files, e.g. results/mbpp_codellama_proactive",
    )
    parser.add_argument(
        "--metric",
        default="pass@1",
        choices=["pass@1", "pass@5", "pass@10", "pass_ratio"],
        help="Metric to average and plot.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom plot title.",
    )
    parser.add_argument(
        "--perturbed_label",
        default="perturbed",
        help="Legend label for perturbed bars.",
    )
    parser.add_argument(
        "--original_label",
        default="original",
        help="Legend label for original bars.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to save the figure.",
    )
    args = parser.parse_args()

    summary = load_scores(args.results_root, args.metric)
    if not summary:
        raise ValueError(f"No eval_results.csv files found under {args.results_root}")

    title = args.title or f"Original vs Perturbed by Perturbation Type ({args.metric})"
    plot_scores(
        summary=summary,
        metric=args.metric,
        title=title,
        output_path=args.output_path,
        perturbed_label=args.perturbed_label,
        original_label=args.original_label,
    )
    print(f"Saved plot to {args.output_path}")


if __name__ == "__main__":
    main()
