import os
import importlib.util
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig
from util.hparams import HyperParams
import re


class ModelLoader:
    @staticmethod
    def _resolve_dtype(hparams: HyperParams, resolved_device: str, config=None):
        dtype_name = getattr(hparams, "dtype", None)
        if dtype_name is None and getattr(hparams, "fp16", False):
            dtype_name = "float16"
        if dtype_name is None and resolved_device.startswith("cuda"):
            config_dtype = getattr(config, "torch_dtype", None)
            if config_dtype is not None:
                return config_dtype
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        if dtype_name is None:
            return torch.float32
        if isinstance(dtype_name, torch.dtype):
            return dtype_name
        dtype_map = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        try:
            return dtype_map[str(dtype_name).lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported dtype: {dtype_name}") from exc

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
            config = AutoConfig.from_pretrained(self.model_name)
            torch_dtype = self._resolve_dtype(hparams, resolved_device, config=config)
            has_accelerate = importlib.util.find_spec("accelerate") is not None
            use_4bit = (
                resolved_device.startswith("cuda")
                and getattr(hparams, "load_in_4bit", False)
                and importlib.util.find_spec("bitsandbytes") is not None
                and has_accelerate
            )
            device_map = {"": resolved_device} if resolved_device.startswith("cuda") and has_accelerate else None
            common_kwargs = {
                "torch_dtype": torch_dtype,
                "low_cpu_mem_usage": True,
            }
            if device_map is not None:
                common_kwargs["device_map"] = device_map
            if use_4bit:
                common_kwargs.pop("torch_dtype", None)
                common_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_quant_type=getattr(hparams, "bnb_4bit_quant_type", "nf4"),
                    bnb_4bit_use_double_quant=getattr(hparams, "bnb_4bit_use_double_quant", True),
                )
            if 'codellama' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, **common_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'qwen' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, **common_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            else:
                raise NotImplementedError
        else:
            self.model, self.tok = self.model_name
        self.model.config.output_hidden_states = True
        if getattr(self.model, "hf_device_map", None) is None and not getattr(self.model, "is_quantized", False):
            self.model.to(resolved_device)
        self.device = resolved_device
        hparams.device = resolved_device
        hparams.dtype = str(torch_dtype).replace("torch.", "") if type(self.model_name) is str else getattr(hparams, "dtype", None)
        self.hparams = hparams
        self.layer_names = [
            n
            for n, m in self.model.named_modules()
            if re.match(r"^(transformer|gpt_neox|model\.layers)\.\d+$", n)
        ]
        self.num_layers = len(self.layer_names)
