import json
from dataclasses import dataclass
from dataclasses import asdict
from typing import List
import yaml


@dataclass
class HyperParams:
    """
    Simple wrapper to store hyperparameters for Python-based rewriting methods.
    """

    @classmethod
    def from_json(cls, fpath):
        with open(fpath, "r") as f:
            data = json.load(f)

        return cls(**data)

    def construct_float_from_scientific_notation(config: dict):
        for key, value in config.items():
            if isinstance(value, str):
                try:
                    # Convert scalar to float if it is in scientific notation format
                    config[key] = float(value)
                except:
                    pass
        return config

    def to_dict(config) -> dict:
        dict = asdict(config)
        return dict


@dataclass
class CREMEHyperParams(HyperParams):
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    rewrite_module_tmp: str
    layer_module_tmp: str
    device: int
    model_name: str

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        return cls(**config)
