import os
from model import ModelLoader
from copy import deepcopy
from typing import Any, Dict, Tuple,Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from util import nethook
from util import CREMEHyperParams
import torch.nn.functional as F
from task_list import TaskList
from util.utils import (
    evaluate_prompt,
    get_problem,
    write_csv_header_if_not_exists,
    append_row_to_csv,
)
from causal_trace import L2_causal_trace    
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def apply_my_knowledge_edit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    ori_prompt,
    pert_prompt,
    hparams: CREMEHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_rep_align_edit(model, tok, ori_prompt,pert_prompt, hparams)

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy

def execute_rep_align_edit(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    orig_prompt: str,
    pert_prompt: str,
    hparams,
    max_new_tokens: int = 32,
) -> Dict[str, torch.Tensor]:
    """
    调整模型使得 target_layer 层中，perturbed prompt 生成部分的隐藏状态更接近 original prompt 的生成部分。
    且original prompt输出的隐藏状态尽可能不变
    """

    device = hparams.device
    layer_name = hparams.layer_module_tmp.format(hparams.layers[0])

    # 获取需要编辑的参数
    weights = {
        n: p
        for n, p in model.named_parameters()
        if any(hparams.rewrite_module_tmp.format(layer) in n for layer in hparams.layers)
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    for name, p in model.named_parameters():
        p.requires_grad = name in weights
    optimizer = torch.optim.Adam(
        [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )

    # 编码原始和扰动 prompt，并手动构造带 continuation 的输入
    with torch.no_grad():
        orig_inputs = tokenizer(orig_prompt, return_tensors="pt").to(device)
        pert_inputs = tokenizer(pert_prompt, return_tensors="pt").to(device)

        # 使用 generate 获取 gold completion，便于构造 prompt + continuation 的完整序列
        orig_gen_ids = model.generate(**orig_inputs, max_new_tokens=max_new_tokens)
        pert_gen_ids = model.generate(**pert_inputs, max_new_tokens=max_new_tokens)

        # 拼接 input_ids：prompt + continuation
        orig_concat = torch.cat([orig_inputs["input_ids"], orig_gen_ids[:, orig_inputs["input_ids"].shape[1]:]], dim=1)
        pert_concat = torch.cat([pert_inputs["input_ids"], orig_gen_ids[:, orig_inputs["input_ids"].shape[1]:]], dim=1)

        # 记录生成 token 起始位置
        orig_start = orig_inputs["input_ids"].shape[1]
        pert_start = pert_inputs["input_ids"].shape[1]

    # 定义 Hook 函数，保存指定层输出
    hidden_states = {}
    def capture_output(x, layer_name_inner):
        if layer_name_inner == layer_name:
            x0 = x[0] if isinstance(x, tuple) else x  # shape: [B, T, D]
            hidden_states["out"] = x0
        return x

    # 获取原始隐藏状态
    with torch.no_grad(), nethook.TraceDict(
        model, [layer_name], edit_output=capture_output
    ):
        model(input_ids=orig_concat)

    orig_hidden = hidden_states["out"][:, orig_start:, :].detach()  # 只保留生成部分
    print("orig_hidden.shape:",orig_hidden.shape)
    # 优化扰动 prompt 生成部分使其靠近原始生成部分
    last_loss = float("inf")
    no_improve_count = 0
    patience = 3  # 可配置，连续几步不下降就停止
    for step in range(hparams.num_steps):
        print(f"== 第 {step+1}/{hparams.num_steps} 步 ==")
        hidden_states.clear()
        optimizer.zero_grad()

        with nethook.TraceDict(
            model, [layer_name], edit_output=capture_output
        ):
            model(input_ids=pert_concat)

        pert_hidden = hidden_states["out"][:, pert_start:, :]  # 只保留生成部分
        print("pert_hidden:",pert_hidden.shape)
        hidden_states.clear()
        with nethook.TraceDict(model, [layer_name], edit_output=capture_output):
            model(input_ids=orig_concat)
        orig_hidden_new = hidden_states["out"][:, orig_start:, :]
        print("ori_hidden_new:",pert_hidden.shape)
        min_len = min(orig_hidden.shape[1], pert_hidden.shape[1])
        min_len_orig = min(orig_hidden.shape[1], orig_hidden_new.shape[1])
        print("min_len_orig:",min_len_orig)
        loss_main = F.mse_loss(pert_hidden[:, :min_len], orig_hidden[:, :min_len])
        loss_reg = F.mse_loss(orig_hidden_new[:, :min_len_orig], orig_hidden[:, :min_len_orig])
        # 总损失
        lambda_reg = hparams.lambda_reg if hasattr(hparams, 'lambda_reg') else 0.1
        loss = loss_main + lambda_reg * loss_reg
        print(f"Loss_main: {loss_main.item():.6f}, Loss_reg: {loss_reg.item():.6f}, Total: {loss.item():.6f}")
        if loss.item() < last_loss:
            last_loss = loss.item()
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= patience:
            print("No improvement in loss, early stopping.")
            break
        loss.backward()
        optimizer.step()

    # 返回 delta 并恢复原始参数
    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]
    print("参数已恢复，返回 delta。")
    return deltas
