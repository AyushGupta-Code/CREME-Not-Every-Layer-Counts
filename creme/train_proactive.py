import os
import sys
import torch
import torch.nn.functional as F
# from torch.cuda.amp import GradScaler, autocast
from torch.amp import GradScaler, autocast

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


def _get_hidden_batch(model, tokenizer, prompts, layer_names, device, no_grad=True, amp_dtype=None):
    """Capture hidden states for one or more layers for a batch of prompts."""
    tokenizer.padding_side = "left"
    tokens = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(device)

    if isinstance(layer_names, str):
        layer_names = [layer_names]
    captured = {}

    def capture_output(x, lname):
        captured[lname] = x[0] if isinstance(x, tuple) else x
        return x

    if no_grad:
        with torch.no_grad():
            if amp_dtype is not None:
                # with autocast(dtype=amp_dtype):
                with autocast("cuda", dtype=amp_dtype):
                    with nethook.TraceDict(model, layer_names, edit_output=capture_output):
                        model(**tokens)
            else:
                with nethook.TraceDict(model, layer_names, edit_output=capture_output):
                    model(**tokens)
    else:
        if amp_dtype is not None:
            # with autocast(dtype=amp_dtype):
            with autocast("cuda", dtype=amp_dtype):
                with nethook.TraceDict(model, layer_names, edit_output=capture_output):
                    model(**tokens)
        else:
            with nethook.TraceDict(model, layer_names, edit_output=capture_output):
                model(**tokens)

    if len(layer_names) == 1:
        return captured[layer_names[0]], tokens
    return captured, tokens


def _masked_mean(hidden, attention_mask):
    """Mean-pool hidden states over non-padding tokens."""
    mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


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


def _resolve_target_module_name(model, target_layer: int) -> str:
    """Return the trainable module path for the selected layer."""
    candidates = [
        f"model.layers.{target_layer}.mlp.down_proj",
        f"base_model.model.model.layers.{target_layer}.mlp.down_proj",
    ]
    module_names = {name for name, _ in model.named_modules()}
    for name in candidates:
        if name in module_names:
            return name
    raise LookupError(
        f"Could not find target module for layer {target_layer}. Tried: {candidates}"
    )


def _parse_target_layers(target_layer: int, target_layers=None):
    """Normalize CLI/YAML layer selection into a non-empty list of ints."""
    if target_layers is None:
        return [target_layer]
    if isinstance(target_layers, str):
        target_layers = [int(layer.strip()) for layer in target_layers.split(",") if layer.strip()]
    return [int(layer) for layer in target_layers]


def run_proactive_finetuning(
    model,
    tokenizer,
    target_layer: int,
    task_name: str,
    save_path: str,
    train_mode: str = "lora",
    target_layers=None,
    lambda_reg: float = 0.01,
    lr: float = 1e-5,
    num_epochs: int = 1,
    smoke_test: bool = False,
    pairs_file: str = None,
    batch_size: int = 4,
    grad_accum_steps: int = 1,
    use_bf16: bool = False,
    use_fp16: bool = False,
):
    """
    Fine-tune model with:
        L_total = L_ce + lambda_reg * L_reg
        L_ce  = cross-entropy on clean prompt
        L_reg = 1 - cosine_similarity(h_pert, h_clean.detach(), dim=-1).mean()

    In `lora` mode, LoRA is applied only to model.layers.{target_layer}.mlp.down_proj.
    In `full` mode, the original down_proj weights at the selected layer(s) are trained directly.
    All other weights are frozen.

    Args:
        model:             The loaded HuggingFace model (already on device).
        tokenizer:         Corresponding tokenizer.
        target_layer:      Index of the causal layer identified in Step 1.
        task_name:         e.g. "mbpp_codellama" — used to select training data.
        save_path:         Directory to save the fine-tuned model after each epoch.
        train_mode:        "lora" for adapter tuning, "full" to train target weights directly.
        target_layers:     Optional list of layer indices to train together; defaults to [target_layer].
        lambda_reg:        Weight for the representation alignment loss.
        num_epochs:        Number of full passes over training pairs.
        smoke_test:        If True, only use 5 pairs and 1 epoch for a quick sanity check.
        batch_size:        Number of pairs per forward pass (default 4).
        grad_accum_steps:  Accumulate gradients over N batches before optimizer step (default 1).
        use_bf16:          Enable bfloat16 mixed precision (recommended for Ampere+).
        use_fp16:          Enable float16 mixed precision (use if bf16 not supported).
    """
    target_layers = _parse_target_layers(target_layer, target_layers)
    device = next(model.parameters()).device
    print(f"Target layers for proactive fine-tuning: {target_layers}")
    print(f"Training mode: {train_mode}")
    print(f"Batch size: {batch_size} | Grad accum steps: {grad_accum_steps} "
          f"| Effective batch: {batch_size * grad_accum_steps}")

    # --- Mixed precision setup ---
    amp_dtype = None
    scaler = None
    if use_bf16 and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        model = model.to(torch.bfloat16)   # native bf16 — avoids fp32 weight / bf16 grad mismatch
        print("Mixed precision: bfloat16 (model cast natively)")
    elif use_fp16:
        amp_dtype = torch.float16
        scaler = GradScaler("cuda")
        print("Mixed precision: float16 (with GradScaler)")
    else:
        print("Mixed precision: disabled (full fp32)")

    if train_mode == "lora":
        if len(target_layers) != 1:
            raise ValueError("LoRA mode currently supports exactly one target layer. Use --train_mode full for multiple layers.")
        try:
            from peft import get_peft_model, LoraConfig
        except ImportError:
            raise ImportError("peft is required for --train_mode lora: pip install peft accelerate")

        # --- Apply LoRA to down_proj at target_layer only ---
        lora_config = LoraConfig(
            target_modules=[f"model.layers.{target_layers[0]}.mlp.down_proj"],
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    elif train_mode == "full":
        target_module_names = [_resolve_target_module_name(model, layer) for layer in target_layers]
        for _, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if any(name.startswith(f"{module_name}.") for module_name in target_module_names):
                param.requires_grad = True
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Training original weights in {target_module_names} ({trainable_params:,} trainable params)")
    else:
        raise ValueError(f"Unsupported train_mode: {train_mode}")
    layer_names = [_resolve_layer_name(model, layer) for layer in target_layers]

    # --- Build training pairs ---
    if pairs_file is not None:
        print(f"Loading training pairs from {pairs_file}...")
        pairs = [(r["ori_prompt"], r["pert_prompt"]) for r in stream_jsonl(pairs_file)]
    else:
        print(f"Building training pairs for {task_name}...")
        pairs = _build_training_pairs(task_name)
    if smoke_test:
        pairs = pairs[:5]
        num_epochs = 1
    print(f"Training pairs: {len(pairs)}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01,
    )

    # Helper: make batches of size batch_size
    def _make_batches(data, size):
        for i in range(0, len(data), size):
            yield data[i: i + size]

    global_step = 0
    num_batches = (len(pairs) + batch_size - 1) // batch_size

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_reg = 0.0

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(_make_batches(pairs, batch_size)):
            ori_prompts = [p[0] for p in batch]
            pert_prompts = [p[1] for p in batch]

            # --- Clean hidden states: no grad (anchor) ---
            h_clean, tokens_clean = _get_hidden_batch(
                model, tokenizer, ori_prompts, layer_names, device,
                no_grad=True, amp_dtype=amp_dtype,
            )

            # --- CE loss on clean prompts (mask padding in labels) ---
            labels = tokens_clean["input_ids"].clone()
            labels[tokens_clean["attention_mask"] == 0] = -100

            if amp_dtype is not None:
                # with autocast(dtype=amp_dtype):
                with autocast("cuda", dtype=amp_dtype):
                    outputs = model(**tokens_clean, labels=labels)
                    L_ce = outputs.loss
            else:
                outputs = model(**tokens_clean, labels=labels)
                L_ce = outputs.loss

            # --- Perturbed hidden states: with grad ---
            h_pert, tokens_pert = _get_hidden_batch(
                model, tokenizer, pert_prompts, layer_names, device,
                no_grad=False, amp_dtype=amp_dtype,
            )

            # --- Representation alignment loss (masked mean pooling) ---
            if isinstance(h_clean, dict):
                reg_losses = []
                for layer_name in layer_names:
                    h_clean_pooled = _masked_mean(h_clean[layer_name].detach().float(), tokens_clean["attention_mask"])
                    h_pert_pooled = _masked_mean(h_pert[layer_name].float(), tokens_pert["attention_mask"])
                    reg_losses.append(1 - F.cosine_similarity(h_pert_pooled, h_clean_pooled, dim=-1).mean())
                L_reg = torch.stack(reg_losses).mean()
            else:
                h_clean_pooled = _masked_mean(h_clean.detach().float(), tokens_clean["attention_mask"])
                h_pert_pooled = _masked_mean(h_pert.float(), tokens_pert["attention_mask"])
                L_reg = 1 - F.cosine_similarity(h_pert_pooled, h_clean_pooled, dim=-1).mean()

            loss = (L_ce + lambda_reg * L_reg) / grad_accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Track unscaled loss for logging
            total_loss += (L_ce.item() + lambda_reg * L_reg.item())
            total_ce += L_ce.item()
            total_reg += L_reg.item()

            # --- Optimizer step every grad_accum_steps batches ---
            is_last_batch = (batch_idx + 1) == num_batches
            if (batch_idx + 1) % grad_accum_steps == 0 or is_last_batch:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if (batch_idx + 1) % 50 == 0 or is_last_batch:
                step_loss = L_ce.item() + lambda_reg * L_reg.item()
                print(
                    f"Epoch {epoch+1} | Batch {batch_idx+1}/{num_batches} | "
                    f"loss={step_loss:.4f} L_ce={L_ce.item():.4f} L_reg={L_reg.item():.4f}"
                )

        n = num_batches
        avg_loss = total_loss / n
        avg_ce = total_ce / n
        avg_reg = total_reg / n
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
    parser.add_argument("--hparams", required=True, help="Path to hparams YAML (e.g. ./creme/hparams/codellama.yaml).")
    parser.add_argument("--task_name", required=True, help="Task name key (e.g. mbpp_codellama, mbpp_qwen).")
    parser.add_argument("--save_path", required=True, help="Directory to save the fine-tuned model checkpoint (e.g. ./models/codellama_proactive_C1_C2_C3).")
    parser.add_argument("--train_mode", choices=["lora", "full"], default="lora",
                        help="Training mode: 'lora' keeps the existing adapter path, 'full' trains the target layer weights directly.")
    parser.add_argument("--target_layers", default=None,
                        help="Optional comma-separated layer list for full tuning, e.g. 28,30. Defaults to target_layer from the hparams YAML.")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate for AdamW (default 1e-5).")
    parser.add_argument("--lambda_reg", type=float, required=True, help="Weight for representation alignment loss (e.g. 0.01).")
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of training epochs.")
    parser.add_argument("--smoke_test", action="store_true", help="Run with 5 pairs and 1 epoch for a quick sanity check.")
    parser.add_argument("--pairs_file", default=None,
                        help="Path to pre-built JSONL pairs file. If set, skips building pairs from scratch.")
    # GPU utilization knobs
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Pairs per forward pass (default 4). Increase to saturate GPU.")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Accumulate gradients over N batches before optimizer step.")
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--bf16", action="store_true",
                           help="Use bfloat16 mixed precision (recommended for RTX 6000 Ada).")
    precision.add_argument("--fp16", action="store_true",
                           help="Use float16 mixed precision with GradScaler.")
    args = parser.parse_args()

    hparams = CREMEHyperParams.from_hparams(args.hparams)
    mt = ModelLoader.from_hparams(hparams)

    run_proactive_finetuning(
        model=mt.model,
        tokenizer=mt.tok,
        target_layer=hparams.target_layer,
        task_name=args.task_name,
        save_path=args.save_path,
        train_mode=args.train_mode,
        target_layers=args.target_layers,
        lambda_reg=args.lambda_reg,
        lr=args.lr,
        num_epochs=args.num_epochs,
        smoke_test=args.smoke_test,
        pairs_file=args.pairs_file,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        use_bf16=args.bf16,
        use_fp16=args.fp16,
    )
