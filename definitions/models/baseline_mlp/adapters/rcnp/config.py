# definitions/models/baseline_mlp/adapters/rcnp/config.py
from dataclasses import dataclass
from MLsystem.utils.config_base import BaseConfig, conf_field

@dataclass
class AdapterConfig(BaseConfig):
  _name: str = "rcnp"
  
  target_dataset: str = conf_field(
    default="rcnp",
    desc="Target dataset",
    ui_mode="hidden",
    ignore=True
  )