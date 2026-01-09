from dataclasses import dataclass
from typing import Dict, Any
import yaml

@dataclass
class Config:
    seed: int
    img_size: int
    batch_size: int
    epochs: int
    lr: float
    conf_th: float

    csv_path: str
    dataset_dir: str
    yolo_model_path: str

    folder_map: Dict[str, str]
    num_lobes: int

    hidden_dim: int
    use_rationale: bool
    rationale_weight: float

    ckpt_path: str

def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        d: Dict[str, Any] = yaml.safe_load(f)
    return Config(**d)
