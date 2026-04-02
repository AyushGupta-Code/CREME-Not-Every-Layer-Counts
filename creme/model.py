import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from util.hparams import HyperParams
import re


class ModelLoader:
    @classmethod
    def from_hparams(cls, hparams: HyperParams):
        return cls(hparams)

    def __init__(self,
                 hparams: HyperParams,
                 ):
        self.model_name = hparams.model_name
        requested_device = getattr(hparams, "device", 0)
        if torch.cuda.is_available():
            resolved_device = f"cuda:{requested_device}"
        else:
            resolved_device = "cpu"
        if type(self.model_name) is str:
            self.model_name = os.path.abspath(self.model_name) if os.path.exists(self.model_name) else self.model_name
            device_map = None
            use_fp16 = (
                hasattr(hparams, "fp16")
                and hparams.fp16
                and resolved_device.startswith("cuda")
            )
            torch_dtype = torch.float16 if use_fp16 else torch.float32
            if 'codellama' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, output_hidden_states=True, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'qwen' in self.model_name.lower():
                self.model = self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, output_hidden_states=True, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            else:
                raise NotImplementedError
        else:
            self.model, self.tok = self.model_name
        self.model.to(resolved_device)
        self.device = resolved_device
        hparams.device = resolved_device
        self.hparams = hparams
        self.layer_names = [
            n
            for n, m in self.model.named_modules()
            if re.match(r"^(transformer|gpt_neox|model\.layers)\.\d+$", n)
        ]
        self.num_layers = len(self.layer_names)
