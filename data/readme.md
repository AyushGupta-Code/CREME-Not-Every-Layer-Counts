# 📁 Dataset

This directory contains the **original and perturbed datasets** used in the CREME experiments.

### 📂 humaneval/

- `original/HumanEval.jsonl`: The standard HumanEval dataset introduced by [Chen et al., 2021].
- `perturbed/{pert_type}.jsonl`: Variants of HumanEval prompts generated using specific perturbation types (e.g., A1, D2, etc.), provided by [Chen et al., 2024] via the **NLPerturbator** tool.

### 📂 mbpp/

- `original/mbpp_original.jsonl`: The original MBPP dataset introduced by [Austin et al., 2021].
- `perturbed/{pert_type}.jsonl`: Perturbed versions of MBPP prompts, also generated using **NLPerturbator**.

## Data Sources

- **HumanEval**:  
  - Source: [Evaluating Large Language Models Trained on Code (Chen et al., 2021)]  
  - Link: https://arxiv.org/pdf/2107.03374

- **MBPP (Mostly Basic Python Problems)**:  
  - Source: [Program Synthesis with Large Language Models (Austin et al., 2021)]  
  - Link: https://arxiv.org/pdf/2108.07732

- **Perturbations**:  
  - Source: [NLPerturbator: Benchmarking LLM Robustness via Naturalistic Prompt Variants (Chen et al., 2024)]  
  - Link: https://dl.acm.org/doi/abs/10.1145/3745764
