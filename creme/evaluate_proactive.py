"""
Standalone evaluation script for the proactively fine-tuned model.

Run from project root:
    python creme/evaluate_proactive.py --task_name mbpp_codellama --pert_type A1 --condition baseline
    python creme/evaluate_proactive.py --task_name mbpp_codellama --all_pert_types --condition baseline
    python creme/evaluate_proactive.py --task_name mbpp_codellama --pert_type A1 --compare
"""

import argparse
import os
import csv
import gc
import sys
import json
import torch

# Allow running from project root with `python creme/evaluate_proactive.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from creme.task_list import TaskList
from creme.util.utils import (
    evaluate_mbpp_prompt,
    get_mbpp_problem,
    build_prompt,
    write_csv_header_if_not_exists,
    append_row_to_csv,
    generate_batch_completion,
    check_correctness_mbpp,
)
from creme.util import CREMEHyperParams
from creme.model import ModelLoader


# All 18 perturbation types used in the mbpp experiments
ALL_PERT_TYPES = [
    "P1", "P2",
    "A1", "A2", "A3",
    "C1", "C2", "C3",
    "D1", "D2", "D3", "D4",
    "E1", "E2", "E3", "E4", "E5", "E6",
]


def load_proactive_model(model_path: str, hparams_path: str):
    """Load proactive model or PEFT adapter by overriding model_name in hparams."""
    hparams = CREMEHyperParams.from_hparams(hparams_path)
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("peft is required to load adapter-only proactive checkpoints") from exc
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config.get("base_model_name_or_path")
        if not base_model_path:
            raise ValueError(f"adapter_config.json in {model_path} is missing base_model_name_or_path")
        hparams.model_name = base_model_path
        editor = ModelLoader.from_hparams(hparams)
        editor.model = PeftModel.from_pretrained(editor.model, model_path)
        editor.model.eval()
        return editor

    hparams.model_name = model_path
    editor = ModelLoader.from_hparams(hparams)
    return editor


def evaluate_one_pert_type(
    editor,
    task_name: str,
    pert_type: str,
    condition: str,
    output_dir: str,
):
    """
    Evaluate the proactive model on all tasks for a single pert_type.

    Returns a list of result dicts for summary reporting.
    """
    task_list_instance = TaskList()
    type_case = task_list_instance.get_task_list(task_name)
    task_list = type_case[pert_type]

    os.makedirs(output_dir, exist_ok=True)
    summary_csv = os.path.join(output_dir, "eval_results.csv")

    write_csv_header_if_not_exists(
        summary_csv,
        ["task_id", "condition", "pass@1", "pass@5", "pass@10", "pass_ratio", "pert_type"],
    )

    results = []

    for task_id in task_list:
        print(f"\n  --- task {task_id} ---")

        # Load problems
        ori_problem = get_mbpp_problem(task_id, "data/mbpp/original/mbpp_original.jsonl")
        pert_problem = get_mbpp_problem(task_id, f"data/mbpp/perturbed/{pert_type}.jsonl")

        ori_prompt = build_prompt(ori_problem)
        pert_prompt = build_prompt(pert_problem)

        # Evaluate on PERTURBED prompt (primary metric)
        print(f"    Evaluating PERTURBED prompt ...")
        acc_pert, passk_pert = evaluate_mbpp_prompt(
            editor.model, editor.tok, pert_prompt, pert_problem,
            batch_size=10, num_iterations=1,
        )
        print(f"    perturbed -> pass_ratio={acc_pert:.3f}  pass@1={passk_pert[0]:.3f}  pass@5={passk_pert[1]:.3f}  pass@10={passk_pert[2]:.3f}")

        append_row_to_csv(
            summary_csv,
            [task_id, condition, passk_pert[0], passk_pert[1], passk_pert[2], acc_pert, pert_type],
        )

        results.append({
            "task_id": task_id,
            "condition": condition,
            "pass@1": passk_pert[0],
            "pass@5": passk_pert[1],
            "pass@10": passk_pert[2],
            "pass_ratio": acc_pert,
            "pert_type": pert_type,
            "prompt_type": "perturbed",
        })

        # Also evaluate on ORIGINAL prompt to check clean performance
        print(f"    Evaluating ORIGINAL prompt ...")
        acc_ori, passk_ori = evaluate_mbpp_prompt(
            editor.model, editor.tok, ori_prompt, ori_problem,
            batch_size=10, num_iterations=1,
        )
        print(f"    original  -> pass_ratio={acc_ori:.3f}  pass@1={passk_ori[0]:.3f}  pass@5={passk_ori[1]:.3f}  pass@10={passk_ori[2]:.3f}")

        append_row_to_csv(
            summary_csv,
            [task_id, f"{condition}_original", passk_ori[0], passk_ori[1], passk_ori[2], acc_ori, pert_type],
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return results


def print_summary_table(all_results):
    """Print a summary table: mean pass@1 on perturbed prompts, grouped by pert_type."""
    from collections import defaultdict

    pert_scores = defaultdict(list)
    for r in all_results:
        if r["prompt_type"] == "perturbed":
            pert_scores[r["pert_type"]].append(r["pass@1"])

    print("\n" + "=" * 55)
    print(f"{'pert_type':<12}  {'n_tasks':>8}  {'mean pass@1 (perturbed)':>22}")
    print("-" * 55)

    grand_total = []
    for pt in sorted(pert_scores.keys()):
        scores = pert_scores[pt]
        mean_p1 = sum(scores) / len(scores) if scores else float("nan")
        grand_total.extend(scores)
        print(f"{pt:<12}  {len(scores):>8}  {mean_p1:>22.4f}")

    if grand_total:
        overall = sum(grand_total) / len(grand_total)
        print("-" * 55)
        print(f"{'OVERALL':<12}  {len(grand_total):>8}  {overall:>22.4f}")
    print("=" * 55 + "\n")


def load_baseline_results(task_name: str, pert_type: str):
    """
    Load baseline edit results from `results/{task_name}/{pert_type}/edit_result.csv`.
    Returns a dict keyed by task_id -> row dict, or None if file not found.
    """
    baseline_csv = os.path.join("results", task_name, pert_type, "edit_result.csv")
    if not os.path.exists(baseline_csv):
        return None

    rows = {}
    with open(baseline_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id_raw = row.get("task_id", "")
            rows[task_id_raw] = row
    return rows


def load_best_creme_result(task_name: str, pert_type: str, task_id: int):
    summary_csv = os.path.join("results", task_name, pert_type, "layer_summary.csv")
    code_json = os.path.join("results", task_name, pert_type, "code_results.json")
    edit_csv = os.path.join("results", task_name, pert_type, "edit_result.csv")
    if not os.path.exists(summary_csv) or not os.path.exists(code_json):
        if not os.path.exists(edit_csv):
            raise FileNotFoundError(
                f"Missing CREME result files under results/{task_name}/{pert_type}/"
            )
        with open(edit_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (
                    str(row.get("task_id")) == str(task_id)
                    and str(row.get("edit_task")) == str(task_id)
                ):
                    return {
                        "layer": "stored_edit",
                        "pass@1": float(row.get("pass@1", "nan")),
                        "pass_ratio": row.get("pass_ratio"),
                        "completion": "",
                        "passed": None,
                        "result": "Detailed completion unavailable; loaded from edit_result.csv",
                    }
        raise ValueError(
            f"No stored edit_result row found for task {task_id} in {edit_csv}"
        )

    best_row = None
    with open(summary_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("task_id")) != str(task_id):
                continue
            layer = row.get("layer", "")
            if layer in {"orig", "pert", ""}:
                continue
            try:
                pass_at_1 = float(row.get("pass@1", "nan"))
            except (TypeError, ValueError):
                continue
            if best_row is None or pass_at_1 > best_row["pass@1"]:
                best_row = {
                    "layer": layer,
                    "pass@1": pass_at_1,
                    "pass_ratio": row.get("pass_ratio"),
                }
    if best_row is None:
        raise ValueError(
            f"No CREME layer rows found for task {task_id} in {summary_csv}"
        )

    with open(code_json) as f:
        entries = json.load(f)

    matching = [
        entry for entry in entries
        if str(entry.get("task_id")) == str(task_id)
        and str(entry.get("layer")) == str(best_row["layer"])
    ]
    if not matching:
        raise ValueError(
            f"No CREME completions found for task {task_id}, layer {best_row['layer']} in {code_json}"
        )

    passing = next((entry for entry in matching if entry.get("passed")), None)
    chosen = passing or matching[0]
    best_row["completion"] = chosen.get("completion", "").strip()
    best_row["passed"] = bool(chosen.get("passed"))
    best_row["result"] = chosen.get("result", {}).get("result") if isinstance(chosen.get("result"), dict) else None
    return best_row


def evaluate_single_task(editor, task_id: int, pert_type: str):
    ori_problem = get_mbpp_problem(task_id, "data/mbpp/original/mbpp_original.jsonl")
    pert_problem = get_mbpp_problem(task_id, f"data/mbpp/perturbed/{pert_type}.jsonl")
    ori_prompt = build_prompt(ori_problem)
    pert_prompt = build_prompt(pert_problem)

    proactive_completion = generate_batch_completion(
        editor.model,
        editor.tok,
        pert_prompt,
        batch_size=1,
    )[0]
    proactive_result = check_correctness_mbpp(
        pert_prompt,
        pert_problem,
        proactive_completion,
        timeout=3.0,
        completion_id=0,
    )
    proactive_metrics = evaluate_mbpp_prompt(
        editor.model,
        editor.tok,
        pert_prompt,
        pert_problem,
        batch_size=10,
        num_iterations=1,
    )
    return {
        "ori_prompt": ori_prompt,
        "pert_prompt": pert_prompt,
        "proactive_completion": proactive_completion.strip(),
        "proactive_result": proactive_result,
        "proactive_pass_ratio": proactive_metrics[0],
        "proactive_passk": proactive_metrics[1],
    }


def print_single_task_compare(results_task_name: str, pert_type: str, task_id: int, proactive_eval: dict):
    creme_result = load_best_creme_result(results_task_name, pert_type, task_id)
    proactive_result = proactive_eval["proactive_result"]

    print("\n" + "=" * 80)
    print(f"Task {task_id} | pert_type={pert_type}")
    print("-" * 80)
    print(
        f"Stored CREME source: {creme_result['layer']} | "
        f"pass@1={creme_result['pass@1']:.4f} | passed_sample={creme_result['passed']}"
    )
    print(
        f"Proactive model: pass@1={proactive_eval['proactive_passk'][0]:.4f} | "
        f"pass_ratio={proactive_eval['proactive_pass_ratio']:.4f} | "
        f"sample_passed={proactive_result['passed']}"
    )
    print("-" * 80)
    print("Stored CREME completion:")
    print(creme_result["completion"] or "<not available in stored results>")
    print("-" * 80)
    print("Proactive completion:")
    print(proactive_eval["proactive_completion"] or "<empty>")
    print("-" * 80)
    print(f"Stored CREME execution result: {creme_result.get('result')}")
    print(f"Proactive execution result: {proactive_result.get('result')}")
    print("=" * 80 + "\n")


def print_comparison_table(all_results, task_name: str):
    """Print side-by-side comparison: proactive model vs baseline edit results."""
    from collections import defaultdict

    # Group proactive results by (pert_type, task_id)
    proactive_map = {}
    for r in all_results:
        if r["prompt_type"] == "perturbed":
            key = (r["pert_type"], str(r["task_id"]))
            proactive_map[key] = r

    pert_types_seen = sorted({r["pert_type"] for r in all_results if r["prompt_type"] == "perturbed"})

    header = f"{'pert_type':<8}  {'task_id':>8}  {'proactive pass@1':>17}  {'baseline pass@1':>16}  {'delta':>7}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for pert_type in pert_types_seen:
        baseline_rows = load_baseline_results(task_name, pert_type)

        keys_for_pt = [(pt, tid) for (pt, tid) in proactive_map if pt == pert_type]
        keys_for_pt.sort(key=lambda x: x[1])

        for key in keys_for_pt:
            _, task_id_str = key
            pro_p1 = proactive_map[key]["pass@1"]

            base_p1 = None
            if baseline_rows is not None:
                # Baseline CSV may use integer or string task_ids; try both
                row = baseline_rows.get(task_id_str) or baseline_rows.get(int(task_id_str) if task_id_str.isdigit() else task_id_str)
                if row is not None:
                    try:
                        base_p1 = float(row.get("pass@1", "nan"))
                    except (ValueError, TypeError):
                        base_p1 = None

            if base_p1 is not None:
                delta = pro_p1 - base_p1
                print(f"{pert_type:<8}  {task_id_str:>8}  {pro_p1:>17.4f}  {base_p1:>16.4f}  {delta:>+7.4f}")
            else:
                print(f"{pert_type:<8}  {task_id_str:>8}  {pro_p1:>17.4f}  {'N/A':>16}  {'N/A':>7}")

    print("=" * len(header) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the proactively fine-tuned model on MBPP perturbations."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/codellama_proactive",
        help="Path to the proactive fine-tuned model (default: ./models/codellama_proactive)",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="mbpp_codellama",
        help="Task name key in TaskList (e.g. mbpp_codellama, mbpp_codellama_proactive)",
    )
    parser.add_argument(
        "--pert_type",
        type=str,
        default=None,
        help="Single perturbation type to evaluate (e.g. A1). Ignored when --all_pert_types is set.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="proactive",
        choices=["causal", "random", "baseline", "proactive"],
        help="Condition label written to the results CSV (default: proactive)",
    )
    parser.add_argument(
        "--all_pert_types",
        action="store_true",
        help="Loop over all 18 perturbation types automatically.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="After evaluation, print a side-by-side comparison against baseline edit results.",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=None,
        help="Evaluate and compare a single MBPP task id.",
    )
    parser.add_argument(
        "--results_task_name",
        type=str,
        default=None,
        help="Task name used to load stored CREME results from results/<task_name>/<pert_type>/.",
    )
    parser.add_argument(
        "--hparams_path",
        type=str,
        default=None,
        help="Path to hparams YAML. Inferred from task_name if not provided.",
    )
    args = parser.parse_args()

    # Infer hparams path
    if args.hparams_path is not None:
        hparams_path = args.hparams_path
    elif "codellama" in args.task_name.lower():
        hparams_path = "./creme/hparams/codellama.yaml"
    elif "qwen" in args.task_name.lower():
        hparams_path = "./creme/hparams/qwen.yaml"
    else:
        raise ValueError(
            f"Cannot infer hparams_path from task_name '{args.task_name}'. "
            "Please provide --hparams_path explicitly."
        )

    results_task_name = args.results_task_name or args.task_name

    # Determine which pert_types to evaluate
    if args.all_pert_types:
        pert_types_to_run = ALL_PERT_TYPES
    elif args.pert_type is not None:
        pert_types_to_run = [args.pert_type]
    else:
        parser.error("Provide either --pert_type <type> or --all_pert_types.")

    print(f"\n{'='*60}")
    print(f"  Proactive model evaluation")
    print(f"  model_path : {args.model_path}")
    print(f"  task_name  : {args.task_name}")
    print(f"  condition  : {args.condition}")
    print(f"  pert_types : {pert_types_to_run}")
    print(f"{'='*60}\n")

    # Load model once — reused across all pert_types
    print(f"Loading model from {args.model_path} ...")
    editor = load_proactive_model(args.model_path, hparams_path)
    print("Model loaded.\n")

    if args.task_id is not None:
        if len(pert_types_to_run) != 1:
            raise ValueError("--task_id requires exactly one perturbation type. Use --pert_type.")
        proactive_eval = evaluate_single_task(
            editor=editor,
            task_id=args.task_id,
            pert_type=pert_types_to_run[0],
        )
        if args.compare:
            print_single_task_compare(
                results_task_name=results_task_name,
                pert_type=pert_types_to_run[0],
                task_id=args.task_id,
                proactive_eval=proactive_eval,
            )
        else:
            print(
                f"task_id={args.task_id} pert_type={pert_types_to_run[0]} "
                f"pass@1={proactive_eval['proactive_passk'][0]:.4f} "
                f"pass_ratio={proactive_eval['proactive_pass_ratio']:.4f} "
                f"sample_passed={proactive_eval['proactive_result']['passed']}"
            )
        del editor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return

    all_results = []

    for pert_type in pert_types_to_run:
        print(f"\n{'='*60}")
        print(f"  pert_type: {pert_type}")
        print(f"{'='*60}")

        output_dir = os.path.join("results", f"{args.task_name}_proactive", pert_type)

        results = evaluate_one_pert_type(
            editor=editor,
            task_name=args.task_name,
            pert_type=pert_type,
            condition=args.condition,
            output_dir=output_dir,
        )
        all_results.extend(results)

        print(f"\n  Results written to: {output_dir}/eval_results.csv")

    # Summary table
    print_summary_table(all_results)

    # Optional comparison against baseline
    if args.compare:
        print_comparison_table(all_results, results_task_name)

    # Cleanup
    del editor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
