"""
Compute per-task average pass@1/5/10 scores, segregated by perturbation type.

Handles two result formats automatically:
  1. Proactive  — results/{run}/*/eval_results.csv
                  columns: task_id, condition, pass@1, pass@5, pass@10, pass_ratio, pert_type
                  groups by: (task_id, condition)

  2. Reactive   — results/{run}/*/edit_result.csv
                  columns: task_id, status, edit_task, pass@1, pass@5, pass@10, pass_ratio
                  groups by: task_id   (averages across all edit_task contexts)

One output CSV is written per perturbation type to:
    <results_dir>/averaged/<pert_type>_avg.csv

Usage:
    python average_scores.py --results_dir results/mbpp_codellama_proactive
    python average_scores.py --results_dir results/mbpp_codellama
    python average_scores.py --results_dir results/mbpp_codellama_proactive --out_dir my_output/
"""

import argparse
import csv
import os
import glob
from collections import defaultdict


METRICS = ["pass@1", "pass@5", "pass@10", "pass_ratio"]


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _detect_format(csv_path: str) -> str:
    """Return 'proactive' or 'reactive' based on the CSV header."""
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f), [])
    if "condition" in header:
        return "proactive"
    if "edit_task" in header:
        return "reactive"
    return "unknown"


# ──────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────

def load_proactive(csv_path: str) -> dict:
    """
    Returns:
        { (task_id_str, condition): {metric: [values], ...} }
    """
    buckets = defaultdict(lambda: {m: [] for m in METRICS})
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            task_id  = str(row.get("task_id", "")).strip()
            condition = str(row.get("condition", "")).strip()
            if not task_id or not condition:
                continue
            key = (task_id, condition)
            for m in METRICS:
                v = _safe_float(row.get(m))
                if v is not None:
                    buckets[key][m].append(v)
    return buckets


def load_reactive(csv_path: str) -> dict:
    """
    Returns:
        { task_id_str: {metric: [values], ...} }
    Averages across all edit_task contexts.
    """
    buckets = defaultdict(lambda: {m: [] for m in METRICS})
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            task_id = str(row.get("task_id", "")).strip()
            if not task_id:
                continue
            for m in METRICS:
                v = _safe_float(row.get(m))
                if v is not None:
                    buckets[task_id][m].append(v)
    return buckets


# ──────────────────────────────────────────────
# Aggregation → rows
# ──────────────────────────────────────────────

def _mean(values: list) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def proactive_to_rows(buckets: dict) -> list[dict]:
    """Convert (task_id, condition) buckets → list of row dicts, sorted."""
    rows = []
    for (task_id, condition), metrics in sorted(buckets.items(), key=lambda x: (int(x[0][0]) if x[0][0].isdigit() else x[0][0], x[0][1])):
        row = {"task_id": task_id, "condition": condition, "n_samples": len(metrics["pass@1"])}
        for m in METRICS:
            row[f"avg_{m}"] = _mean(metrics[m])
        rows.append(row)
    return rows


def reactive_to_rows(buckets: dict) -> list[dict]:
    """Convert task_id buckets → list of row dicts, sorted."""
    rows = []
    for task_id, metrics in sorted(buckets.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        row = {"task_id": task_id, "n_edit_tasks": len(metrics["pass@1"])}
        for m in METRICS:
            row[f"avg_{m}"] = _mean(metrics[m])
        rows.append(row)
    return rows


# ──────────────────────────────────────────────
# Writer
# ──────────────────────────────────────────────

def write_csv(path: str, rows: list[dict], fieldnames: list[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def process_results_dir(results_dir: str, out_dir: str):
    # Find all pert_type subdirectories that contain a known CSV
    pert_dirs = sorted([
        d for d in glob.glob(os.path.join(results_dir, "*"))
        if os.path.isdir(d)
    ])

    if not pert_dirs:
        print(f"No subdirectories found under '{results_dir}'.")
        return

    processed = 0

    for pert_dir in pert_dirs:
        pert_type = os.path.basename(pert_dir)

        # Locate the CSV — prefer eval_results.csv, fall back to edit_result.csv
        csv_path = None
        fmt = None
        for candidate, candidate_fmt in [
            (os.path.join(pert_dir, "eval_results.csv"), None),
            (os.path.join(pert_dir, "edit_result.csv"),  None),
        ]:
            if os.path.exists(candidate):
                csv_path = candidate
                fmt = _detect_format(candidate)
                break

        if csv_path is None or fmt == "unknown":
            print(f"  [{pert_type}] no recognised CSV found — skipping")
            continue

        out_path = os.path.join(out_dir, f"{pert_type}_avg.csv")

        if fmt == "proactive":
            buckets = load_proactive(csv_path)
            rows    = proactive_to_rows(buckets)
            fields  = ["task_id", "condition", "n_samples",
                       "avg_pass@1", "avg_pass@5", "avg_pass@10", "avg_pass_ratio"]
        else:  # reactive
            buckets = load_reactive(csv_path)
            rows    = reactive_to_rows(buckets)
            fields  = ["task_id", "n_edit_tasks",
                       "avg_pass@1", "avg_pass@5", "avg_pass@10", "avg_pass_ratio"]

        write_csv(out_path, rows, fields)
        print(f"  [{pert_type}]  {fmt:10s}  {len(rows):3d} rows  ->  {out_path}")
        processed += 1

    print(f"\nDone. {processed} perturbation type(s) written to: {out_dir}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Average per-task pass@k scores by perturbation type."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Root results directory, e.g. results/mbpp_codellama_proactive  or  results/mbpp_codellama",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for averaged CSVs. Defaults to <results_dir>/averaged/",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_dir, "averaged")
    print(f"\nResults dir : {args.results_dir}")
    print(f"Output dir  : {out_dir}\n")

    process_results_dir(args.results_dir, out_dir)


if __name__ == "__main__":
    main()
