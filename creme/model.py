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
        if type(self.model_name) is str:
            device_map = None
            torch_dtype = torch.float16 if hasattr(
                hparams, 'fp16') and hparams.fp16 else torch.float32
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
        self.model.to(f'cuda:{hparams.device}')
        self.hparams = hparams
        self.layer_names = [
            n
            for n, m in self.model.named_modules()
            if re.match(r"^(transformer|gpt_neox|model\.layers)\.\d+$", n)
        ]
        self.num_layers = len(self.layer_names)
