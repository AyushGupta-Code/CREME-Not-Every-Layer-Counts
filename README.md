# CREME: Code Robustness Enhancement via Model Editing
A model editing method to enhance the robustness of codeLLMs.

This repository provides the replication package for our ICSE submission titled: **CREME: Robustness Enhancement of Code LLMs via Layer-Aware Model Editing**

##  Environment Setup
We recommend using Python 3.10+ with CUDA-compatible GPU. You can install the dependencies via:
```Python
conda create -n creme python==3.10
conda activate creme
pip install -r requirements.txt
```

## Repository Structure
```Python
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

## Running an Experiment
The main script is main.py, which performs the following steps:
1. Selects an editing task of a given perturbation type (e.g., “A1”)
2. Locates robustness-sensitive layers using L2-based causal tracing
3. Applies parameter-level knowledge editing
4. Evaluates the edited model on the entire category of tasks

### Example (MBPP + CodeLLaMA, perturbation type A1):
```Python
python main.py
```
The default configuration in main.py runs:
```Python
model_editing("A1", "mbpp_codellama")
```
To run other configurations, modify the last line in main.py, e.g.:
```Python
model_editing("D2", "humaneval_qwen")
model_editing("E1", "humaneval_codellama")
model_editing("S2", "mbpp_qwen")
```
## Output Format
Results are written to:
```Python
results/{task_name}/{pert_type}/edit_result.csv
```