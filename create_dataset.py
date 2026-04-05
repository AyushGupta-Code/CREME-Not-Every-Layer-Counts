"""
Create training pairs dataset for C1, C2, C3 perturbation types.

Output: data/training_pairs_C1_C2_C3.jsonl
Each line: {"task_id": int, "pert_type": str, "ori_prompt": str, "pert_prompt": str}
"""

import os
import sys
import json

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from creme.util.utils import load_sanitized_mbpp, get_mbpp_problem, build_prompt

PERT_TYPES = ["C1", "C2", "C3"]

ORIG_PATH = os.path.join(project_root, "data", "mbpp", "original", "mbpp_original.jsonl")
PERT_DIR = os.path.join(project_root, "data", "mbpp", "perturbed")
OUT_PATH = os.path.join(project_root, "data", "training_pairs_C1_C2_C3.jsonl")


def build_pairs():
    problems = load_sanitized_mbpp(ORIG_PATH)
    pairs = []
    skipped = 0

    for task_id, problem in problems.items():
        ori_prompt = build_prompt(problem)
        for pert_type in PERT_TYPES:
            pert_path = os.path.join(PERT_DIR, f"{pert_type}.jsonl")
            if not os.path.exists(pert_path):
                print(f"[WARN] Perturbation file not found: {pert_path}")
                continue
            try:
                pert_problem = get_mbpp_problem(task_id, pert_path)
                pert_prompt = build_prompt(pert_problem)
                pairs.append({
                    "task_id": task_id,
                    "pert_type": pert_type,
                    "ori_prompt": ori_prompt,
                    "pert_prompt": pert_prompt,
                })
            except (KeyError, StopIteration):
                skipped += 1

    return pairs, skipped


def main():
    print(f"Loading MBPP from: {ORIG_PATH}")
    pairs, skipped = build_pairs()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"Written {len(pairs)} pairs to: {OUT_PATH}")
    if skipped:
        print(f"Skipped {skipped} task/pert_type combinations (task_id not found in perturbed file)")

    # Summary
    from collections import Counter
    counts = Counter(p["pert_type"] for p in pairs)
    for pt in PERT_TYPES:
        print(f"  {pt}: {counts.get(pt, 0)} pairs")


if __name__ == "__main__":
    main()
