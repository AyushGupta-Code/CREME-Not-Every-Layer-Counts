
from util.utils import (
    evaluate_prompt,
    write_csv_header_if_not_exists,
    append_row_to_csv,
    filter_code,
    fix_indents,
    check_correctness,
    append_json_record,
    pass_at_k,
    check_correctness_mbpp,
    build_prompt,
    evaluate_mbpp_prompt
)
from util import nethook
import torch


def layername(model, num, kind=None):
    if hasattr(model, "model") and hasattr(model.model, "layers"):  # likely LLaMA structure
        if kind == "embed":
            return "model.embed_tokens"
        if kind == "attn":
            kind = "self_attn"
        return f"model.layers.{num}{'' if kind is None else '.' + kind}"


def locate_toxic_layer(model, tokenizer, ori_prompt, pert_prompt, layers):
    tokenizer.padding_side = 'left'
    device = next(model.parameters()).device
    input = tokenizer([ori_prompt, pert_prompt], return_tensors="pt",
                      padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**input, output_hidden_states=True)
    # List of (batch_size=2, seq_len, hidden_dim)
    hidden_states = outputs.hidden_states
    max_layer = None
    max_l2 = float('-inf')
    for layer in layers:
        if layer + 1 >= len(hidden_states):
            print(f"Warning: layer {layer} exceeds available layers.")
            continue
        h_ori = hidden_states[layer + 1][0]
        h_pert = hidden_states[layer + 1][1]
        l2_dist = torch.norm(h_ori - h_pert, p=2).item()
        print(f"Layer {layer}: L2 diff = {l2_dist:.4f}")
        if l2_dist > max_l2:
            max_l2 = l2_dist
            max_layer = layer
    print(f"\nKey layer is Layer {max_layer}, the L2 diff is {max_l2:.4f}")
    return max_layer


def L2_causal_trace(mt, task_id, dic_path, pert_type, ori_problem, pert_problem, batch_size=5, num_iterations=1):
    pert_prompt = pert_problem["prompt"].replace("    ", "\t")
    orig_prompt = ori_problem["prompt"].replace("    ", "\t")
    problem = ori_problem
    summary_csv = dic_path+'/layer_summary.csv'
    code_json = dic_path+'/code_results.json'
    acc_orig, passk1 = evaluate_prompt(
        mt.model, mt.tok, orig_prompt, problem, batch_size, num_iterations)
    acc_pert, passk2 = evaluate_prompt(
        mt.model, mt.tok, pert_prompt, problem, batch_size, num_iterations)
    print(
        f"original pass@1: {acc_orig:.2f} | perturbed pass@1: {acc_pert:.2f}")
    write_csv_header_if_not_exists(summary_csv, [
                                   "task_id", "layer", "pass@1", "pass@5", "pass@10", "pass_ratio", "improvement"])
    append_row_to_csv(
        summary_csv, [problem["task_id"], "orig", *passk1, acc_orig, None])
    append_row_to_csv(
        summary_csv, [problem["task_id"], "pert", *passk2, acc_pert, None])
    print("\n====================")
    print(f"Start intervention HumanEval/{task_id} [{pert_type}]...")
    print("====================")
    tokenizer = mt.tok
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    input_batch = tokenizer(
        [orig_prompt] + [pert_prompt] * batch_size,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(mt.model.device)
    input_ids_cutoff = input_batch.input_ids.size(dim=1)
    improvements = []
    best_layer = []
    best_restore = 0.0
    best_pass_rate_layer = []
    best_pass_rate = 0
    for layer in range(mt.num_layers):
        print(f"\nIntervention layer {layer} ...")

        def patch_hidden(x, layer_name):
            if layer_name == layername(mt.model, layer):
                if isinstance(x, tuple):
                    x0 = x[0]
                    if x0.shape[1] > 1:
                        for i in range(1, batch_size + 1):
                            x0[i, -1, :] = x0[0, -1, :].detach().clone()
                    return (x0,) + x[1:]
                else:
                    if x.shape[1] > 1:
                        for i in range(1, batch_size + 1):
                            x[i, -1, :] = x[0, -1, :].detach().clone()
                    return x
            return x

        with torch.no_grad(), nethook.TraceDict(
            mt.model,
            [layername(mt.model, layer)],
            edit_output=patch_hidden
        ):
            out = mt.model.generate(
                input_ids=input_batch["input_ids"],
                attention_mask=input_batch["attention_mask"],
                max_new_tokens=128,
                do_sample=True,
                temperature=0.2,
                top_p=0.95,
                pad_token_id=mt.tok.pad_token_id,
            )

            pass_count = 0
            for i in range(1, batch_size + 1):
                completion = mt.tok.decode(
                    out[i][input_ids_cutoff:], skip_special_tokens=True)
                code = filter_code(fix_indents(completion))
                result = check_correctness(
                    problem, code, timeout=3.0, completion_id=i)
                if result["passed"]:
                    pass_count += 1
                append_json_record(code_json, {
                    "task_id": problem["task_id"],
                    "layer": layer,
                    "sample_id": i,
                    "completion": code,
                    "result": result,
                    "passed": result["passed"]
                })
            c_orig = pass_count
            ratio = c_orig / batch_size if batch_size > 0 else 0.0
            k_list = [1, 5, 10]
            passk = [pass_at_k(batch_size, c_orig, k) for k in k_list]

            print(f"Layer {layer} pass@1-5-10: {passk}, ratio: {ratio:.2f}")
            if ratio > best_pass_rate:
                best_pass_rate = ratio
                best_pass_rate_layer = []
                best_pass_rate_layer.append(layer)
            elif ratio == best_pass_rate:
                best_pass_rate_layer.append(layer)
            improvement = 0
            if acc_orig - acc_pert > 0:
                improvement = (ratio - acc_pert) / (acc_orig - acc_pert)
                improvement = max(improvement, 0)
            if improvement > best_restore:
                best_restore = improvement
                best_layer = []
                best_layer.append(layer)
            elif improvement == best_restore:
                best_layer.append(layer)
            improvements.append(improvement)
            append_row_to_csv(
                summary_csv, [problem["task_id"], layer, *passk, ratio, improvement])
    if best_restore == 0:
        best_layer = best_pass_rate_layer
    print(
        f"\nCompleted intervention, the results are saved to {summary_csv} and {code_json}")
    print(
        f"The key layer is layer {best_layer}; Restoration Improvement:{best_restore:.2f}")
    key_layer = locate_toxic_layer(
        mt.model, mt.tok, orig_prompt, pert_prompt, best_layer)
    return key_layer


# mbpp
def mbpp_L2_causal_trace(mt, task_id, dic_path, pert_type, ori_problem, pert_problem, batch_size=5, num_iterations=1):
    pert_prompt = build_prompt(pert_problem)
    orig_prompt = build_prompt(ori_problem)
    problem = ori_problem
    summary_csv = dic_path+'/layer_summary.csv'
    code_json = dic_path+'/code_results.json'

    acc_orig, passk1 = evaluate_mbpp_prompt(
        mt.model, mt.tok, orig_prompt, problem, batch_size, num_iterations)
    acc_pert, passk2 = evaluate_mbpp_prompt(
        mt.model, mt.tok, pert_prompt, pert_problem, batch_size, num_iterations)
    print(
        f"original pass@1: {acc_orig:.2f} | perturbed pass@1: {acc_pert:.2f}")
    write_csv_header_if_not_exists(summary_csv, [
                                   "task_id", "layer", "pass@1", "pass@5", "pass@10", "pass_ratio", "improvement"])
    append_row_to_csv(
        summary_csv, [problem["task_id"], "orig", *passk1, acc_orig, None])
    append_row_to_csv(
        summary_csv, [problem["task_id"], "pert", *passk2, acc_pert, None])
    print("\n====================")
    print(f"Start intervention mbpp:{task_id} [{pert_type}]...")
    print("====================")

    tokenizer = mt.tok
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_batch = tokenizer(
        [orig_prompt.replace("    ", "\t")] +
        [pert_prompt.replace("    ", "\t")] * batch_size,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(mt.model.device)
    input_ids_cutoff = input_batch.input_ids.size(dim=1)
    improvements = []
    best_layer = []
    best_restore = 0.0
    best_pass_rate_layer = []
    best_pass_rate = 0
    for layer in range(mt.num_layers):
        print(f"\nIntervention layer {layer} ...")

        def patch_hidden(x, layer_name):
            if layer_name == layername(mt.model, layer):
                if isinstance(x, tuple):
                    x0 = x[0]
                    if x0.shape[1] > 1:
                        for i in range(1, batch_size + 1):
                            x0[i, -1, :] = x0[0, -1, :].detach().clone()
                    return (x0,) + x[1:]
                else:
                    if x.shape[1] > 1:
                        for i in range(1, batch_size + 1):
                            x[i, -1, :] = x[0, -1, :].detach().clone()
                    return x
            return x

        with torch.no_grad(), nethook.TraceDict(
            mt.model,
            [layername(mt.model, layer)],
            edit_output=patch_hidden
        ):
            out = mt.model.generate(
                input_ids=input_batch["input_ids"],
                attention_mask=input_batch["attention_mask"],
                max_new_tokens=128,
                do_sample=True,
                temperature=0.2,
                top_p=0.95,
                pad_token_id=mt.tok.pad_token_id,
            )

            pass_count = 0
            for i in range(1, batch_size + 1):
                completion = mt.tok.decode(
                    out[i][input_ids_cutoff:], skip_special_tokens=True)
                code = filter_code(fix_indents(completion))
                result = check_correctness_mbpp(
                    pert_prompt, pert_problem, code, timeout=3.0, completion_id=i)
                if result["passed"]:
                    pass_count += 1
                append_json_record(code_json, {
                    "task_id": problem["task_id"],
                    "layer": layer,
                    "sample_id": i,
                    "completion": code,
                    "result": result,
                    "passed": result["passed"]
                })

            c_orig = pass_count
            ratio = c_orig / batch_size if batch_size > 0 else 0.0
            k_list = [1, 5, 10]
            passk = [pass_at_k(batch_size, c_orig, k) for k in k_list]

            print(f"Layer {layer} pass@1-5-10: {passk}, ratio: {ratio:.2f}")
            if ratio > best_pass_rate:
                best_pass_rate = ratio
                best_pass_rate_layer = []
                best_pass_rate_layer.append(layer)
            elif ratio == best_pass_rate:
                best_pass_rate_layer.append(layer)
            improvement = 0
            if acc_orig - acc_pert > 0:
                improvement = (ratio - acc_pert) / (acc_orig - acc_pert)
                improvement = max(improvement, 0)
            if improvement > best_restore:
                best_restore = improvement
                best_layer = []
                best_layer.append(layer)
            elif improvement == best_restore:
                best_layer.append(layer)
            improvements.append(improvement)
            append_row_to_csv(
                summary_csv, [problem["task_id"], layer, *passk, ratio, improvement])
    if best_restore == 0:
        best_layer = best_pass_rate_layer
    print(
        f"\nCompleted intervention, the results are saved to {summary_csv} and {code_json}")
    print(
        f"The key layer is layer {best_layer}; Restoration Improvement: {best_restore:.2f}")
    key_layer = locate_toxic_layer(
        mt.model, mt.tok, orig_prompt, pert_prompt, best_layer)
    return key_layer
