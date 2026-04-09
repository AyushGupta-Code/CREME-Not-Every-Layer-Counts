import argparse
import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt


PERT_TYPES = [
    "A1", "A2", "A3",
    "C1", "C2", "C3",
    "D1", "D2", "D3", "D4",
    "E1", "E2", "E3", "E4", "E5", "E6",
    "P1", "P2",
    "S1", "S2",
]


def _read_average_metric(averaged_dir: str, metric: str, condition: str = None) -> Dict[str, float]:
    """Read one averaged metric per perturbation type from *_avg.csv files."""
    scores = {}
    for pert_type in PERT_TYPES:
        csv_path = os.path.join(averaged_dir, f"{pert_type}_avg.csv")
        if not os.path.exists(csv_path):
            continue

        values: List[float] = []
        with open(csv_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if metric not in row:
                    raise KeyError(f"Metric '{metric}' not found in {csv_path}")
                if condition is not None and row.get("condition") != condition:
                    continue
                values.append(float(row[metric]))

        if values:
            scores[pert_type] = sum(values) / len(values)

    return scores


def _make_grouped_bar_plot(
    baseline_scores: Dict[str, float],
    candidate_scores: Dict[str, float],
    baseline_label: str,
    candidate_label: str,
    metric: str,
    output_path: str,
):
    pert_types = [pt for pt in PERT_TYPES if pt in baseline_scores or pt in candidate_scores]
    baseline_values = [baseline_scores.get(pt, 0.0) for pt in pert_types]
    candidate_values = [candidate_scores.get(pt, 0.0) for pt in pert_types]

    positions = list(range(len(pert_types)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar([x - width / 2 for x in positions], baseline_values, width=width, label=baseline_label, color="#4C78A8")
    ax.bar([x + width / 2 for x in positions], candidate_values, width=width, label=candidate_label, color="#F58518")

    ax.set_title(f"Averaged Perturbation Performance Comparison ({metric})")
    ax.set_xlabel("Perturbation Type")
    ax.set_ylabel(metric)
    ax.set_xticks(positions)
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
        description="Create a grouped bar chart comparing averaged perturbation-type results between two runs."
    )
    parser.add_argument(
        "--baseline_dir",
        default=os.path.join("results", "mbpp_codellama", "averaged"),
        help="Directory containing baseline *_avg.csv files.",
    )
    parser.add_argument(
        "--candidate_dir",
        default=os.path.join("results", "mbpp_codellama_proactive", "averaged"),
        help="Directory containing candidate *_avg.csv files.",
    )
    parser.add_argument(
        "--baseline_label",
        default="creme",
        help="Legend label for the baseline bars.",
    )
    parser.add_argument(
        "--candidate_label",
        default="codellama_proactive_full_top2_28_30",
        help="Legend label for the candidate bars.",
    )
    parser.add_argument(
        "--metric",
        default="avg_pass@1",
        choices=["avg_pass@1", "avg_pass@5", "avg_pass@10", "avg_pass_ratio"],
        help="Averaged metric to compare.",
    )
    parser.add_argument(
        "--candidate_condition",
        default="proactive",
        help="Condition value to keep from the candidate averaged CSVs.",
    )
    parser.add_argument(
        "--output_path",
        default=os.path.join("results", "comparison_plots", "creme_vs_codellama_proactive_full_top2_28_30_avg_pass1.png"),
        help="Path to save the output plot.",
    )
    args = parser.parse_args()

    baseline_scores = _read_average_metric(args.baseline_dir, args.metric)
    candidate_scores = _read_average_metric(args.candidate_dir, args.metric, condition=args.candidate_condition)

    if not baseline_scores:
        raise ValueError(f"No baseline scores found in {args.baseline_dir}")
    if not candidate_scores:
        raise ValueError(
            f"No candidate scores found in {args.candidate_dir} for condition '{args.candidate_condition}'"
        )

    _make_grouped_bar_plot(
        baseline_scores=baseline_scores,
        candidate_scores=candidate_scores,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        metric=args.metric,
        output_path=args.output_path,
    )

    print(f"Saved comparison plot to {args.output_path}")


if __name__ == "__main__":
    main()
