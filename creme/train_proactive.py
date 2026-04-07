import csv
import json
import os
import sys
from collections import Counter

import torch
import torch.nn.functional as F

# Ensure creme/ is on path (mirrors utils.py pattern)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from util import nethook
from util.utils import build_prompt, load_sanitized_mbpp, get_mbpp_problem, stream_jsonl

PERT_TYPES = [
    "A1", "A2", "A3",
    "C1", "C2", "C3",
    "D1", "D2", "D3", "D4",
    "E1", "E2", "E3", "E4", "E5", "E6",
    "S1", "S2",
]


def _resolve_pairs_file(pairs_file: str) -> str:
    if pairs_file is None or os.path.exists(pairs_file):
        return pairs_file
    root, ext = os.path.splitext(pairs_file)
    candidates = []
    if ext == ".json":
        candidates.append(root + ".jsonl")
    elif ext == ".jsonl":
        candidates.append(root + ".json")
    else:
        candidates.extend([pairs_file + ".jsonl", pairs_file + ".json"])
    for candidate in candidates:
        if os.path.exists(candidate):
            print(f"Pairs file {pairs_file} not found; using {candidate} instead.")
            return candidate
    raise FileNotFoundError(f"Could not find pairs file: {pairs_file}")


def _build_training_pairs(task_name: str) -> list:
    """
    Build (ori_prompt, pert_prompt) pairs from all MBPP tasks × all pert_types.
    Skips silently when a task_id is not found in a perturbed file.
    """
    if "mbpp" in task_name:
        orig_path = "data/mbpp/original/mbpp_original.jsonl"
        pert_dir = "data/mbpp/perturbed"
    else:
        raise ValueError(f"Proactive training only supports mbpp tasks, got: {task_name}")

    problems = load_sanitized_mbpp(orig_path)
    pairs = []
    for task_id, problem in problems.items():
        ori_prompt = build_prompt(problem)
        for pert_type in PERT_TYPES:
            pert_path = os.path.join(pert_dir, f"{pert_type}.jsonl")
            if not os.path.exists(pert_path):
                continue
            try:
                pert_problem = get_mbpp_problem(task_id, pert_path)
                pert_prompt = build_prompt(pert_problem)
                pairs.append((ori_prompt, pert_prompt))
            except (KeyError, StopIteration):
                continue
    return pairs


def _group_training_pairs(pairs: list) -> list:
    grouped = {}
    for ori_prompt, pert_prompt in pairs:
        grouped.setdefault(ori_prompt, [])
        grouped[ori_prompt].append(pert_prompt)
    return [(ori_prompt, pert_prompts) for ori_prompt, pert_prompts in grouped.items()]


def _infer_model_and_dataset(task_name: str):
    task_name_lower = task_name.lower()
    if "codellama" in task_name_lower:
        model_label = "CodeLLaMA"
    elif "qwen" in task_name_lower:
        model_label = "Qwen"
    else:
        raise ValueError(f"Could not infer model label from task_name={task_name}")

    if "mbpp" in task_name_lower:
        dataset_label = "MBPP"
    elif "humaneval" in task_name_lower:
        dataset_label = "HumanEval"
    else:
        raise ValueError(f"Could not infer dataset label from task_name={task_name}")

    return model_label, dataset_label


def _load_top_layers_from_results(task_name: str, results_csv: str, top_k: int) -> list:
    model_label, dataset_label = _infer_model_and_dataset(task_name)
    counts = Counter()

    with open(results_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("model") != model_label or row.get("dataset") != dataset_label:
                continue
            try:
                counts[int(row["key_layer"])] += 1
            except (TypeError, ValueError, KeyError):
                continue

    if not counts:
        raise ValueError(
            f"No key-layer rows found in {results_csv} for model={model_label}, dataset={dataset_label}"
        )

    top_layers = [layer for layer, _ in counts.most_common(top_k)]
    print(
        f"Top {top_k} layers from {results_csv} for {model_label}/{dataset_label}: "
        f"{top_layers}"
    )
    return top_layers


def _resolve_layer_name(model, target_layer: int) -> str:
    """Return the hookable module path for a transformer block."""
    candidates = [
        f"model.layers.{target_layer}",
        f"base_model.model.model.layers.{target_layer}",
    ]
    module_names = {name for name, _ in model.named_modules()}
    for name in candidates:
        if name in module_names:
            return name
    raise LookupError(
        f"Could not find target layer {target_layer}. Tried: {candidates}"
    )


def _resolve_target_layers(model, target_layers=None, target_layer=None) -> list:
    base_layers = getattr(getattr(model, "model", None), "layers", None)
    if base_layers is None and hasattr(model, "base_model"):
        base_layers = getattr(getattr(getattr(model.base_model, "model", None), "model", None), "layers", None)
    if base_layers is None:
        raise ValueError("Could not resolve the model layer stack")

    num_layers = len(base_layers)
    selected = target_layers if target_layers else [target_layer]
    if not selected or selected == [None]:
        raise ValueError("No target layers were provided")

    resolved = []
    for layer in selected:
        if layer is None:
            continue
        layer = int(layer)
        if layer < 0:
            layer = num_layers + layer
        if layer < 0 or layer >= num_layers:
            raise ValueError(f"Resolved target layer {layer} is out of range [0, {num_layers - 1}]")
        if layer not in resolved:
            resolved.append(layer)
    if not resolved:
        raise ValueError("No valid target layers were resolved")
    return resolved


def _get_hidden_dict(model, tokenizer, prompt, layer_names, device, no_grad=True):
    """Capture hidden states at the requested transformer blocks."""
    tokens = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512, padding=True
    ).to(device)

    captured = {}

    def capture_output(x, lname):
        captured[lname] = x[0] if isinstance(x, tuple) else x
        return x

    if no_grad:
        with torch.no_grad(), nethook.TraceDict(model, layer_names, edit_output=capture_output):
            model(**tokens)
    else:
        with nethook.TraceDict(model, layer_names, edit_output=capture_output):
            model(**tokens)

    return captured, tokens


def _move_tokens_to_device(tokens, device):
    return {key: value.to(device) for key, value in tokens.items()}


def _pretokenize_grouped_pairs(tokenizer, grouped_pairs: list) -> list:
    prepared = []
    for ori_prompt, pert_prompts in grouped_pairs:
        tokens_clean = tokenizer(
            ori_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        tokens_pert = tokenizer(
            pert_prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        prepared.append((tokens_clean, tokens_pert))
    return prepared


def _get_hidden_from_tokens(model, tokens, layer_names, no_grad=True):
    captured = {}

    def capture_output(x, lname):
        captured[lname] = x[0] if isinstance(x, tuple) else x
        return x

    if no_grad:
        with torch.no_grad(), nethook.TraceDict(model, layer_names, edit_output=capture_output):
            model(**tokens)
    else:
        with nethook.TraceDict(model, layer_names, edit_output=capture_output):
            model(**tokens)

    return captured


def _freeze_all_but_selected_layers(model, target_layers: list):
    selected_prefixes = tuple(
        f"{prefix}{layer}."
        for layer in target_layers
        for prefix in ("model.layers.", "base_model.model.model.layers.")
    )

    for _, param in model.named_parameters():
        param.requires_grad = False

    trainable_names = []
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if name.startswith(selected_prefixes):
            param.requires_grad = True
            trainable_names.append(name)
            trainable_params += param.numel()

    if trainable_params == 0:
        raise ValueError(f"No trainable parameters matched target layers {target_layers}")

    print(f"Direct fine-tuning transformer layers: {target_layers}")
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    return trainable_names


def _write_training_metadata(save_path: str, task_name: str, target_layers: list):
    os.makedirs(save_path, exist_ok=True)
    metadata = {
        "task_name": task_name,
        "target_layers": target_layers,
    }
    metadata_path = os.path.join(save_path, "training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved training metadata to {metadata_path}")
    return metadata_path


def run_proactive_finetuning(
    model,
    tokenizer,
    target_layer: int = None,
    target_layers: list = None,
    task_name: str = None,
    save_path: str = None,
    lambda_reg: float = 0.01,
    num_epochs: int = 1,
    smoke_test: bool = False,
    pairs_file: str = None,
):
    """
    Fine-tune the selected transformer blocks directly with:
        L_total = L_ce + lambda_reg * L_reg
        L_ce  = cross-entropy on clean prompt
        L_reg = average_i(1 - cosine_similarity(h_pert_i, h_clean_i.detach()))

    All weights are frozen except the selected transformer layers.
    No LoRA adapters are used.
    """
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        raise ValueError(
            "Direct layer fine-tuning cannot run on a 4-bit/8-bit loaded base model. "
            "Load the model in full precision and try again."
        )

    device = next(model.parameters()).device
    target_layers = _resolve_target_layers(
        model,
        target_layers=target_layers,
        target_layer=target_layer,
    )
    layer_names = [_resolve_layer_name(model, layer) for layer in target_layers]

    print(f"Target layers for direct fine-tuning + regularization: {target_layers}")
    _write_training_metadata(save_path, task_name, target_layers)
    model.config.use_cache = False
    if str(device).startswith("cuda") and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if str(device).startswith("cuda") and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    _freeze_all_but_selected_layers(model, target_layers)

    # --- Build training pairs ---
    if pairs_file is not None:
        pairs_file = _resolve_pairs_file(pairs_file)
        print(f"Loading training pairs from {pairs_file}...")
        pairs = [(r["ori_prompt"], r["pert_prompt"]) for r in stream_jsonl(pairs_file)]
    else:
        print(f"Building training pairs for {task_name}...")
        pairs = _build_training_pairs(task_name)
    if smoke_test:
        pairs = pairs[:5]
        num_epochs = 1
    grouped_pairs = _group_training_pairs(pairs)
    print(f"Training pairs: {len(pairs)}")
    print(f"Unique clean prompts / optimization steps per epoch: {len(grouped_pairs)}")
    print("Pre-tokenizing training prompts...")
    tokenized_grouped_pairs = _pretokenize_grouped_pairs(tokenizer, grouped_pairs)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5,
        weight_decay=0.01,
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_reg = 0.0

        for step, (tokens_clean_cpu, tokens_pert_cpu) in enumerate(tokenized_grouped_pairs):
            optimizer.zero_grad(set_to_none=True)

            tokens_clean = _move_tokens_to_device(tokens_clean_cpu, device)
            tokens_pert = _move_tokens_to_device(tokens_pert_cpu, device)

            hidden_clean = _get_hidden_from_tokens(
                model, tokens_clean, layer_names, no_grad=True
            )

            outputs = model(**tokens_clean, labels=tokens_clean["input_ids"])
            l_ce = outputs.loss

            hidden_pert = _get_hidden_from_tokens(
                model, tokens_pert, layer_names, no_grad=False
            )

            reg_terms = []
            for layer_name in layer_names:
                h_clean = hidden_clean[layer_name]
                h_pert = hidden_pert[layer_name]
                reg_terms.append(
                    1 - F.cosine_similarity(
                        h_pert.mean(dim=1),
                        h_clean.detach().mean(dim=1),
                        dim=-1,
                    ).mean()
                )
            l_reg = torch.stack(reg_terms).mean()

            loss = l_ce + lambda_reg * l_reg
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ce += l_ce.item()
            total_reg += l_reg.item()

            if (step + 1) % 50 == 0:
                print(
                    f"Epoch {epoch+1} | Step {step+1}/{len(grouped_pairs)} | "
                    f"loss={loss.item():.4f} L_ce={l_ce.item():.4f} L_reg={l_reg.item():.4f}"
                )

        avg_loss = total_loss / len(grouped_pairs)
        avg_ce = total_ce / len(grouped_pairs)
        avg_reg = total_reg / len(grouped_pairs)
        print(
            f"Epoch {epoch+1} complete | avg_loss={avg_loss:.4f} "
            f"avg_L_ce={avg_ce:.4f} avg_L_reg={avg_reg:.4f}"
        )

        epoch_path = os.path.join(save_path, f"epoch_{epoch+1}")
        os.makedirs(epoch_path, exist_ok=True)
        model.save_pretrained(epoch_path)
        tokenizer.save_pretrained(epoch_path)
        print(f"Checkpoint saved to {epoch_path}")

    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Final model saved to {save_path}")

    return model


if __name__ == "__main__":
    import argparse
    from model import ModelLoader
    from util.hparams import CREMEHyperParams

    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", default="./creme/hparams/codellama.yaml")
    parser.add_argument("--task_name", default="mbpp_codellama")
    parser.add_argument("--save_path", default="./models/codellama_proactive")
    parser.add_argument("--lambda_reg", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument(
        "--pairs_file",
        default=None,
        help="Path to pre-built JSONL pairs file (e.g. data/training_pairs_C1_C2_C3.jsonl). If set, skips building pairs from scratch.",
    )
    parser.add_argument(
        "--target_layers",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of transformer layers to fine-tune directly, e.g. --target_layers 28 30 8",
    )
    parser.add_argument(
        "--top_k_from_results",
        type=int,
        default=None,
        help="Automatically select the top-k most frequent key layers from --results_csv for the given task_name.",
    )
    parser.add_argument(
        "--results_csv",
        default="results/key_layer/key_layer_results.csv",
        help="CSV used with --top_k_from_results to derive target layers.",
    )
    args = parser.parse_args()

    hparams = CREMEHyperParams.from_hparams(args.hparams)
    if args.top_k_from_results is not None:
        resolved_target_layers = _load_top_layers_from_results(
            task_name=args.task_name,
            results_csv=args.results_csv,
            top_k=args.top_k_from_results,
        )
    elif args.target_layers is not None:
        resolved_target_layers = args.target_layers
    elif getattr(hparams, "target_layers", None):
        resolved_target_layers = hparams.target_layers
    else:
        resolved_target_layers = [hparams.target_layer]

    print(f"Selected target layers: {resolved_target_layers}")

    mt = ModelLoader.from_hparams(hparams)

    run_proactive_finetuning(
        model=mt.model,
        tokenizer=mt.tok,
        target_layer=hparams.target_layer,
        target_layers=resolved_target_layers,
        task_name=args.task_name,
        save_path=args.save_path,
        lambda_reg=args.lambda_reg,
        num_epochs=args.num_epochs,
        smoke_test=args.smoke_test,
        pairs_file=args.pairs_file,
    )
