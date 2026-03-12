# CREME: Code Robustness Enhancement via Model Editing
A model editing method to enhance the robustness of codeLLMs.

This repository provides the replication package for our ICSE submission titled:**CREME: Robustness Enhancement of Code LLMs via Layer-Aware Model Editing**

## Environment Setup
We recommend Python 3.10+ with a CUDA-compatible GPU.

### Move to shared scratch space
Use your first group from `groups` as `group_name`. For this project, the shared scratch path is:
```bash
cd /share/csc591008s26/agupta86
git clone https://github.com/AyushGupta-Code/CREME-Not-Every-Layer-Counts.git
cd CREME-Not-Every-Layer-Counts
```

If the repository is already present, just move into it:
```bash
cd /share/csc591008s26/agupta86/CREME-Not-Every-Layer-Counts
```

### Create a Python environment
This repository currently provides a `requirements.txt`, so a virtual environment is sufficient:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Repository Structure
```text
.
├── main.py                        # Entry point for running CREME experiments
├── creme/
│   ├── task_list.py              # Task groupings by perturbation type
│   ├── util/
│   │   ├── utils.py              # Evaluation utilities
│   │   ├── hparams.py            # Hyperparameter parser
│   │   └── nethook.py            # Layer access and intervention hooks
│   ├── causal_trace.py           # Causal tracing for layer localization
│   ├── edit.py                   # Knowledge editing module
│   ├── model.py                  # Model loading and setup
│   └── hparams/
│       ├── codellama.yaml        # Hyperparameters for CodeLLaMA
│       └── qwen.yaml             # Hyperparameters for QwenCoder
├── data/
│   ├── humaneval/                # Original and perturbed HumanEval data
│   └── mbpp/                     # Original and perturbed MBPP data
├── results/                      # Output folder for experiment results
└── requirements.txt              # Required Python packages
```

## Model Setup
Download the required model weights into the `models/` directory. This code expects:
```text
./models/codellama
./models/qwencoder
```

The model path should match the configuration in `creme/hparams/codellama.yaml` or `creme/hparams/qwen.yaml`, for example:
```yaml
model_name: "./models/codellama"
```

### Configure Hugging Face cache in shared scratch
Large model downloads can exceed your home-directory quota if cache files are written under `~/.cache`. Point all temporary and Hugging Face cache data to `/share` before downloading:
```bash
mkdir -p /share/csc591008s26/agupta86/tmp
mkdir -p /share/csc591008s26/agupta86/hf-cache
mkdir -p /share/csc591008s26/agupta86/hf-home
mkdir -p ./models/codellama ./models/qwencoder

export TMPDIR=/share/csc591008s26/agupta86/tmp
export HF_HOME=/share/csc591008s26/agupta86/hf-home
export HF_HUB_CACHE=/share/csc591008s26/agupta86/hf-cache
export XDG_CACHE_HOME=/share/csc591008s26/agupta86/hf-home
export TRANSFORMERS_CACHE=/share/csc591008s26/agupta86/hf-home/transformers
```

### Download QwenCoder
Qwen is public and can be downloaded without authentication:
```bash
./.venv/bin/hf download Qwen/Qwen2.5-Coder-7B \
  --local-dir ./models/qwencoder
```

### Download CodeLLaMA
CodeLLaMA is gated on Hugging Face. Your account must already have access approved.
```bash
./.venv/bin/hf auth login
./.venv/bin/hf download meta-llama/CodeLlama-7b-hf \
  --local-dir ./models/codellama
```

To log out after downloading:
```bash
./.venv/bin/hf auth logout
```

## Running an Experiment
The main script is main.py, which performs the following steps:
1. Selects an editing task of a given perturbation type (e.g., “A1”)
2. Locates robustness-sensitive layers using L2-based causal tracing
3. Applies parameter-level knowledge editing
4. Evaluates the edited model on the entire category of tasks

### Example (MBPP + CodeLLaMA, perturbation type A1):
```bash
python main.py
```
The default configuration in main.py runs:
```python
model_editing("A1", "mbpp_codellama")
```
To run other configurations, modify the last line in main.py, e.g.:
```python
model_editing("D2", "humaneval_qwen")
model_editing("E1", "humaneval_codellama")
model_editing("S2", "mbpp_qwen")
```
## Output Format
Results are written to:
```text
results/{task_name}/{pert_type}/edit_result.csv
```
