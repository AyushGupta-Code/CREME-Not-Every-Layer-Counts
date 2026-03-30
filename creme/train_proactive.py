import os
import sys
import torch
import torch.nn.functional as F

# Ensure creme/ is on path (mirrors utils.py pattern)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from util import nethook
from util.utils import build_prompt, load_sanitized_mbpp, get_mbpp_problem

PERT_TYPES = [
    "A1", "A2", "A3",
    "C1", "C2", "C3",
    "D1", "D2", "D3", "D4",
    "E1", "E2", "E3", "E4", "E5", "E6",
    "S1", "S2",
]


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


def _get_hidden(model, tokenizer, prompt, layer_name, device, no_grad=True):
    """Capture hidden state at layer_name for the given prompt."""
    tokens = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to(device)

    captured = {}

    def capture_output(x, lname):
        captured["out"] = x[0] if isinstance(x, tuple) else x
        return x

    if no_grad:
        with torch.no_grad(), nethook.TraceDict(model, [layer_name], edit_output=capture_output):
            model(**tokens)
    else:
        with nethook.TraceDict(model, [layer_name], edit_output=capture_output):
            model(**tokens)

    return captured["out"], tokens


def _resolve_layer_name(model, target_layer: int) -> str:
    """Return the hookable module path for a transformer block after wrapping."""
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


def run_proactive_finetuning(
    model,
    tokenizer,
    target_layer: int,
    task_name: str,
    save_path: str,
    lambda_reg: float = 0.01,
    num_epochs: int = 1,
    smoke_test: bool = False,
):
    """
    Fine-tune model with:
        L_total = L_ce + lambda_reg * L_reg
        L_ce  = cross-entropy on clean prompt
        L_reg = 1 - cosine_similarity(h_pert, h_clean.detach(), dim=-1).mean()

    LoRA is applied only to model.layers.{target_layer}.mlp.down_proj.
    All other weights are frozen.

    Args:
        model:        The loaded HuggingFace model (already on device).
        tokenizer:    Corresponding tokenizer.
        target_layer: Index of the causal layer identified in Step 1.
        task_name:    e.g. "mbpp_codellama" — used to select training data.
        save_path:    Directory to save the fine-tuned model after each epoch.
        lambda_reg:   Weight for the representation alignment loss.
        num_epochs:   Number of full passes over training pairs.
        smoke_test:   If True, only use 5 pairs and 1 epoch for a quick sanity check.
    """
    try:
        from peft import get_peft_model, LoraConfig
    except ImportError:
        raise ImportError("peft is required: pip install peft accelerate")

    device = next(model.parameters()).device
    print(f"Target layer for LoRA + regularization: {target_layer}")

    # --- Apply LoRA to down_proj at target_layer only ---
    lora_config = LoraConfig(
        target_modules=[f"model.layers.{target_layer}.mlp.down_proj"],
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    layer_name = _resolve_layer_name(model, target_layer)

    # --- Build training pairs ---
    print(f"Building training pairs for {task_name}...")
    pairs = _build_training_pairs(task_name)
    if smoke_test:
        pairs = pairs[:5]
        num_epochs = 1
    print(f"Training pairs: {len(pairs)}")

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

        for step, (ori_prompt, pert_prompt) in enumerate(pairs):
            optimizer.zero_grad()

            # --- Clean hidden state: no grad (anchor) ---
            h_clean, tokens_clean = _get_hidden(
                model, tokenizer, ori_prompt, layer_name, device, no_grad=True
            )

            # --- CE loss on clean prompt ---
            outputs = model(**tokens_clean, labels=tokens_clean["input_ids"])
            L_ce = outputs.loss

            # --- Perturbed hidden state: with grad ---
            h_pert, _ = _get_hidden(
                model, tokenizer, pert_prompt, layer_name, device, no_grad=False
            )

            # --- Representation alignment loss ---
            # .detach() on h_clean is non-negotiable: prevents collapse
            L_reg = 1 - F.cosine_similarity(
                h_pert.mean(dim=1),
                h_clean.detach().mean(dim=1),
                dim=-1,
            ).mean()

            loss = L_ce + lambda_reg * L_reg
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ce += L_ce.item()
            total_reg += L_reg.item()

            if (step + 1) % 50 == 0:
                print(
                    f"Epoch {epoch+1} | Step {step+1}/{len(pairs)} | "
                    f"loss={loss.item():.4f} L_ce={L_ce.item():.4f} L_reg={L_reg.item():.4f}"
                )

        avg_loss = total_loss / len(pairs)
        avg_ce = total_ce / len(pairs)
        avg_reg = total_reg / len(pairs)
        print(
            f"Epoch {epoch+1} complete | avg_loss={avg_loss:.4f} "
            f"avg_L_ce={avg_ce:.4f} avg_L_reg={avg_reg:.4f}"
        )

        # Save after each epoch
        epoch_path = os.path.join(save_path, f"epoch_{epoch+1}")
        os.makedirs(epoch_path, exist_ok=True)
        model.save_pretrained(epoch_path)
        tokenizer.save_pretrained(epoch_path)
        print(f"Checkpoint saved to {epoch_path}")

    # Save final merged checkpoint
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
    args = parser.parse_args()

    hparams = CREMEHyperParams.from_hparams(args.hparams)
    mt = ModelLoader.from_hparams(hparams)

    run_proactive_finetuning(
        model=mt.model,
        tokenizer=mt.tok,
        target_layer=hparams.target_layer,
        task_name=args.task_name,
        save_path=args.save_path,
        lambda_reg=args.lambda_reg,
        num_epochs=args.num_epochs,
        smoke_test=args.smoke_test,
    )
