# CREME: Code Robustness Enhancement via Model Editing

A model editing method to enhance the robustness of code LLMs.

This repository provides the replication package for our ICSE submission: **CREME: Robustness Enhancement of Code LLMs via Layer-Aware Model Editing**

---

## Environment Setup

Requires Python 3.10+ and a CUDA-compatible GPU.

```bash
conda create -n creme python==3.10
conda activate creme
pip install -r requirements.txt
```

## Setup & Installation

For Windows users, use the following steps to get Python and the Hugging Face CLI set up correctly before downloading models.

1. Ensure Python 3.10 is installed and available:
```powershell
python --version
```
2. Install `huggingface_hub` for your user account:
```powershell
pip install huggingface_hub --user
```
3. Add the user Scripts folder to your `PATH` permanently in PowerShell.
Replace `<your-username>` with your Windows username:
```powershell
[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";C:\Users\<your-username>\AppData\Roaming\Python\Python310\Scripts", "User")

$env:PATH = [Environment]::GetEnvironmentVariable("PATH", "User") + ";" + [Environment]::GetEnvironmentVariable("PATH", "Machine")
```
4. Close and reopen PowerShell.
5. Verify the CLI is available:
```powershell
hf -h
```
6. Log in to Hugging Face and enter your token from `https://huggingface.co/settings/tokens`:
```powershell
hf auth login
```

Note: As of `huggingface_hub` version `1.8.0`, the CLI command changed from `huggingface-cli` to `hf`.

## Common Commands

The Hugging Face CLI now uses `hf` instead of `huggingface-cli`.

| Old command | New command |
| --- | --- |
| `huggingface-cli login` | `hf auth login` |
| `huggingface-cli logout` | `hf auth logout` |
| `huggingface-cli download` | `hf download` |
| `huggingface-cli upload` | `hf upload` |
| `huggingface-cli cache` | `hf cache` |

---

## Model Setup

Download CodeLLaMA-7B and/or QwenCoder-7B and place them under `models/`:

```
models/
├── codellama/     # CodeLLaMA-7B weights
└── qwencoder/     # QwenCoder-7B weights
```

The path must match `model_name` in the corresponding hparams file:

```yaml
# creme/hparams/codellama.yaml
model_name: "./models/codellama"

# creme/hparams/qwen.yaml
model_name: "./models/qwencoder"
```

---

## Repository Structure

```
.
├── main.py                         # Entry point — reactive CREME + proactive fine-tuning
├── creme/
│   ├── causal_trace.py             # L2 causal tracing for layer localization (Step 1)
│   ├── edit.py                     # Reactive knowledge editing (Step 2, reactive)
│   ├── train_proactive.py          # Proactive fine-tuning script (Step 2, proactive)
│   ├── evaluate_proactive.py       # Evaluation script for proactive model
│   ├── model.py                    # Model loading
│   ├── task_list.py                # Task IDs grouped by dataset and perturbation type
│   ├── hparams/
│   │   ├── codellama.yaml          # Hparams for CodeLLaMA (target_layer: 28)
│   │   └── qwen.yaml               # Hparams for QwenCoder
│   └── util/
│       ├── utils.py                # Evaluation, code execution, data loading
│       ├── hparams.py              # HyperParams dataclass
│       └── nethook.py              # Layer hooking (TraceDict)
├── data/
│   ├── humaneval/
│   │   ├── original/HumanEval.jsonl
│   │   └── perturbed/{pert_type}.jsonl
│   └── mbpp/
│       ├── original/mbpp_original.jsonl
│       └── perturbed/{pert_type}.jsonl
├── results/                        # All experiment outputs
└── requirements.txt
```

**Supported perturbation types:** P1, P2, A1–A3, C1–C3, D1–D4, E1–E6, S1–S2

**Supported configurations:** `humaneval_codellama`, `humaneval_qwen`, `mbpp_codellama`, `mbpp_qwen`

---

## Approach 1 — Reactive CREME (original)

Per-task pipeline: locate the causal layer → apply knowledge editing → evaluate.

### Step 1: Configure the run

Edit the last line of `main.py`:

```python
model_editing("A1", "mbpp_codellama")
```

| Argument | Options |
|----------|---------|
| `pert_type` | `P1`, `P2`, `A1`–`A3`, `C1`–`C3`, `D1`–`D4`, `E1`–`E6`, `S1`, `S2` |
| `task_name` | `mbpp_codellama`, `mbpp_qwen`, `humaneval_codellama`, `humaneval_qwen` |

### Step 2: Run

```bash
python main.py
```

### Output

```
results/{task_name}/{pert_type}/
├── edit_result.csv      # pass@1/5/10 and pass_ratio per task, after editing
├── layer_summary.csv    # per-layer restoration improvement scores
└── code_results.json    # generated completions and pass/fail per layer
```

---

## Approach 2 — Proactive Fine-tuning (new)

Train the model once with a representation alignment regularizer so it is robust to all perturbation types without per-task editing at inference time.

**Loss:** `L_total = L_ce + λ · L_reg`
- `L_ce` — cross-entropy on the clean prompt
- `L_reg = 1 − cosine_similarity(h_pert, h_clean.detach())` at the causal layer

### Step 1: Confirm the causal layer

The causal layer for CodeLLaMA-7B on MBPP is pre-set to **layer 28** in `creme/hparams/codellama.yaml`. To rediscover it empirically, run the reactive pipeline on 15–20 tasks and take the mode of the returned key layers.

### Step 2: Run proactive fine-tuning (standalone)

Choose a descriptive `--save_path` for your model — the name does not need to follow any convention and won't affect evaluation.

```bash
# CodeLLaMA on MBPP
python creme/train_proactive.py \
    --hparams ./creme/hparams/codellama.yaml \
    --task_name mbpp_codellama \
    --save_path ./models/<your_run_name> \
    --lambda_reg 0.01 \
    --num_epochs 5

# QwenCoder on MBPP
python creme/train_proactive.py \
    --hparams ./creme/hparams/qwen.yaml \
    --task_name mbpp_qwen \
    --save_path ./models/<your_run_name> \
    --lambda_reg 0.01 \
    --num_epochs 5
```

For a quick sanity check before a full run (only `--smoke_test` has a default — it's a flag):

```bash
python creme/train_proactive.py \
    --hparams ./creme/hparams/codellama.yaml \
    --task_name mbpp_codellama \
    --save_path ./models/<your_run_name> \
    --lambda_reg 0.01 \
    --num_epochs 1 \
    --smoke_test
```

The fine-tuner saves a LoRA adapter (not full weights) to `--save_path`. Checkpoints are written after each epoch to `--save_path/epoch_N/`. The `evaluate_proactive.py` script automatically detects and loads LoRA adapters — no merging step required.

**Or**, run via the integrated pipeline — `main.py` automatically triggers proactive fine-tuning once after the causal layer is found for the first task:

```bash
python main.py   # fine-tuning runs once, then reactive editing proceeds as usual
```

### Step 3: Evaluate the proactive model

Pass `--model_path` to any saved model or LoRA adapter directory. The script infers the correct hparams from the path or `--task_name` automatically. Use `--output_tag` to label the results directory for the run.

```bash
# Single perturbation type — CodeLLaMA
python creme/evaluate_proactive.py \
    --model_path ./models/<your_run_name> \
    --task_name mbpp_codellama \
    --hparams_path ./creme/hparams/codellama.yaml \
    --condition proactive \
    --output_tag proactive \
    --pert_type A1

# All 18 perturbation types — CodeLLaMA
python creme/evaluate_proactive.py \
    --model_path ./models/<your_run_name> \
    --task_name mbpp_codellama \
    --hparams_path ./creme/hparams/codellama.yaml \
    --condition proactive \
    --output_tag proactive \
    --all_pert_types

# With side-by-side comparison against reactive CREME baseline
python creme/evaluate_proactive.py \
    --model_path ./models/<your_run_name> \
    --task_name mbpp_codellama \
    --hparams_path ./creme/hparams/codellama.yaml \
    --condition proactive \
    --output_tag proactive \
    --all_pert_types \
    --compare

# QwenCoder
python creme/evaluate_proactive.py \
    --model_path ./models/<your_qwen_run_name> \
    --task_name mbpp_qwen \
    --hparams_path ./creme/hparams/qwen.yaml \
    --condition proactive \
    --output_tag proactive \
    --all_pert_types
```

**Key CLI arguments:**

| Argument | Required | Description |
|---|---|---|
| `--model_path` | Yes | Path to fine-tuned model or LoRA adapter directory |
| `--task_name` | Yes | Task key (e.g. `mbpp_codellama`, `mbpp_qwen`) |
| `--hparams_path` | Yes | Path to hparams YAML |
| `--condition` | Yes | Label written to CSV for this run (e.g. `proactive`, `random`) |
| `--output_tag` | Yes | Suffix for the results directory (e.g. `proactive`) |
| `--pert_type` | * | Single pert type to evaluate (e.g. `A1`) |
| `--all_pert_types` | * | Evaluate all 18 pert types |
| `--compare` | No | Print side-by-side vs reactive baseline after eval |

*One of `--pert_type` or `--all_pert_types` is required.

### Output

```
results/{task_name}_{output_tag}/{pert_type}/
└── eval_results.csv     # task_id, condition, pass@1/5/10, pass_ratio, pert_type
                         # includes both perturbed and original prompt rows
```

A summary table is printed to stdout showing mean pass@1 per perturbation type.

---

## Ablation Study

To validate that the causal layer (not arbitrary regularization) drives the improvement, train three conditions:

| Condition | Description | Command |
|-----------|-------------|---------|
| C1: Baseline | Reactive CREME results | Already in `results/` after Step 1 |
| C2: Random layer | Proactive fine-tuning at a non-causal layer | See below |
| C3: Causal layer | Proactive fine-tuning at the identified causal layer | Step 2 above |

**Train C2 (random layer):**

```bash
# Temporarily set target_layer: 3 in the hparams YAML, then:
python creme/train_proactive.py \
    --hparams ./creme/hparams/codellama.yaml \
    --task_name mbpp_codellama \
    --save_path ./models/<model_name>_random_layer \
    --lambda_reg 0.01 \
    --num_epochs 5
```

**Evaluate C2:**

```bash
python creme/evaluate_proactive.py \
    --model_path ./models/<model_name>_random_layer \
    --task_name mbpp_codellama \
    --hparams_path ./creme/hparams/codellama.yaml \
    --condition random \
    --output_tag random \
    --all_pert_types \
    --compare
```

If C3 mean pass@1 > C2 mean pass@1, the hypothesis is confirmed: causal layer specificity drives robustness improvement.

---

## Quick Reference

| Goal | Command |
|------|---------|
| Reactive CREME, MBPP + CodeLLaMA, pert A1 | `python main.py` (set last line) |
| Proactive fine-tuning, full run | `python creme/train_proactive.py --hparams <yaml> --task_name <task> --save_path <path> --lambda_reg 0.01 --num_epochs 5` |
| Proactive fine-tuning, smoke test | `python creme/train_proactive.py --hparams <yaml> --task_name <task> --save_path <path> --lambda_reg 0.01 --num_epochs 1 --smoke_test` |
| Evaluate proactive model, single pert type | `python creme/evaluate_proactive.py --model_path <path> --task_name <task> --hparams_path <yaml> --condition proactive --output_tag proactive --pert_type A1` |
| Evaluate proactive model, all pert types | `python creme/evaluate_proactive.py --model_path <path> --task_name <task> --hparams_path <yaml> --condition proactive --output_tag proactive --all_pert_types` |
| Evaluate + compare vs baseline | `python creme/evaluate_proactive.py --model_path <path> --task_name <task> --hparams_path <yaml> --condition proactive --output_tag proactive --all_pert_types --compare` |


## Download Codellama from hf
```
 hf download meta-llama/CodeLlama-7b-hf --local-dir ./CodeLlama-7b
 ```