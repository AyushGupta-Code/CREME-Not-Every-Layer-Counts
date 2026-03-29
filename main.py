
from creme.task_list import TaskList
from creme.util.utils import (
    evaluate_prompt,
    get_problem,
    write_csv_header_if_not_exists,
    append_row_to_csv,
    get_mbpp_problem,
    build_prompt,
    evaluate_mbpp_prompt
)
from creme.util import CREMEHyperParams
from creme.model import ModelLoader
from creme.causal_trace import L2_causal_trace,mbpp_L2_causal_trace
from creme.edit import apply_my_knowledge_edit_to_model
from creme.train_proactive import run_proactive_finetuning
import torch
import os
import gc

def model_editing(pert_type,task_name):
    task_list_instance = TaskList()
    type_case = task_list_instance.get_task_list(task_name)
    task_list=type_case[pert_type]
    test_list=task_list
    print(f"============================start deal type : {pert_type}============================")
    print(f"===============task_list : {task_list}=================")
    print("====================================================================================")
    dic_path=f"results/{task_name}/{pert_type}"
    summary_csv=f'results/{task_name}/{pert_type}/edit_result.csv'
    os.makedirs(dic_path, exist_ok=True)
    write_csv_header_if_not_exists(summary_csv, ["task_id", "status","edit_task", "pass@1", "pass@5", "pass@10", "pass_ratio"])
    proactive_done = False
    proactive_save_path = None
    for task in task_list:
        print(f"\n=== start task {task} ===")
        if "codellama" in task_name:
            hparams_path = "./creme/hparams/codellama.yaml"
        elif "qwen" in task_name:
            hparams_path = "./creme/hparams/qwen.yaml"
        hparams = CREMEHyperParams.from_hparams(hparams_path)
        editor = ModelLoader.from_hparams(hparams)
        if "humaneval" in task_name:
            task_id=f"HumanEval/{task}"
            ori_problem=get_problem(task_id,"data/humaneval/original/HumanEval.jsonl")
            pert_problem=get_problem(task_id,f"data/humaneval/perturbed/{pert_type}.jsonl")
            orig_prompt=ori_problem["prompt"].replace("    ", "\t")
            pert_prompt=pert_problem["prompt"].replace("    ", "\t")
            if len(editor.hparams.layers) == 0:
                key_layer=L2_causal_trace(editor,task,dic_path,pert_type,ori_problem,pert_problem,batch_size=10, num_iterations=1)
                editor.hparams.layers = [key_layer]
                print("find key layer:",key_layer)
            edited_model, weights_copy = apply_my_knowledge_edit_to_model(
                editor.model,
                editor.tok,
                ori_problem["prompt"].replace("    ", "\t"),
                pert_problem["prompt"].replace("    ", "\t"),
                editor.hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=True,
            )
            acc_edit,passk3=evaluate_prompt(edited_model,editor.tok,pert_problem["prompt"],ori_problem)
            print("edited_result:",acc_edit,passk3)
            append_row_to_csv(summary_csv, [task_id,"edit",task_id, *passk3, acc_edit])
            for task2 in test_list:
                if task2==task:
                    continue
                task_id2=f"HumanEval/{task2}"
                pert_problem2=get_problem(task_id2,f"data/humaneval/perturbed/{pert_type}.jsonl")
                acc_edit2,passk4=evaluate_prompt(edited_model,editor.tok,pert_problem2["prompt"],pert_problem2)
                append_row_to_csv(summary_csv, [task_id2,"edit",task_id, *passk4, acc_edit2])
                print(f"task{task2} result after edit:",acc_edit2,passk4)
        elif "mbpp" in task_name:
            task_id=task
            ori_problem=get_mbpp_problem(task_id,"data/mbpp/original/mbpp_original.jsonl")
            pert_problem=get_mbpp_problem(task_id,f"data/mbpp/perturbed/{pert_type}.jsonl")
            ori_prompt = build_prompt(ori_problem)
            pert_prompt = build_prompt(pert_problem)
            # print("ori_prompt:",ori_prompt)
            # print("pert_prompt:",pert_prompt)
            if len(editor.hparams.layers) == 0:
                key_layer=mbpp_L2_causal_trace(editor,task,dic_path,pert_type,ori_problem,pert_problem,batch_size=10, num_iterations=1)
                editor.hparams.layers = [key_layer]
                print("find key layer:",key_layer)
            if not proactive_done:
                target_layer = editor.hparams.layers[0]
                model_type = "codellama" if "codellama" in task_name else "qwen"
                proactive_save_path = f"./models/{model_type}_proactive"
                print(f"\n=== Running proactive fine-tuning at layer {target_layer} ===")
                run_proactive_finetuning(
                    model=editor.model,
                    tokenizer=editor.tok,
                    target_layer=target_layer,
                    task_name=task_name,
                    save_path=proactive_save_path,
                )
                proactive_done = True
                print(f"=== Proactive fine-tuning complete, model saved to {proactive_save_path} ===\n")
            edited_model, weights_copy = apply_my_knowledge_edit_to_model(
                editor.model,
                editor.tok,
                ori_prompt.replace("    ", "\t"),
                pert_prompt.replace("    ", "\t"),
                editor.hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=True,
            )
            acc_edit,passk3=evaluate_mbpp_prompt(edited_model,editor.tok,pert_prompt,ori_problem)
            print("edited_result:",acc_edit,passk3)
            append_row_to_csv(summary_csv, [task_id,"edit",task_id, *passk3, acc_edit])
            for task2 in test_list:
                if task2==task:
                    continue
                task_id2=task2
                pert_problem2=get_mbpp_problem(task_id2,f"data/mbpp/perturbed/{pert_type}.jsonl")
                pert_prompt2=build_prompt(pert_problem2)
                acc_edit2,passk4=evaluate_mbpp_prompt(edited_model,editor.tok,pert_prompt2,pert_problem2)
                append_row_to_csv(summary_csv, [task_id2,"edit",task_id, *passk4, acc_edit2])
                print(f"task{task2} result after edit:",acc_edit2,passk4)
        del editor
        del edited_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print(f"=== clear {task} ===")

if __name__ == "__main__":
    model_editing("A1", "mbpp_codellama")
