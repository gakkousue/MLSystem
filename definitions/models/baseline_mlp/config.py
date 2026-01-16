# definitions/models/baseline_mlp/config.py
from dataclasses import dataclass
from MLsystem.utils.config_base import BaseConfig, conf_field

@dataclass
class ModelConfig(BaseConfig):
  _name: str = "baseline_mlp"

  n_units: int = conf_field(
    default=100,
    desc="隠れ層のユニット数",
    ignore=False
  )
  
  lr: float = conf_field(
    default=0.01,
    desc="学習率",
    ignore=False
  )
  
  step_size: int = conf_field(
    default=200,
    desc="学習率減衰のステップ数(epoch)",
    ignore=False
  )