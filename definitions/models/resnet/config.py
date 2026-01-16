# definitions/models/resnet/config.py
# モデル固有設定（Model Settings）

from dataclasses import dataclass
from MLsystem.utils.config_base import BaseConfig, conf_field

@dataclass
class ModelConfig(BaseConfig):
  _name: str = "resnet"

  num_layers: int = conf_field(
    default=18,
    desc="ResNetの層数 (18, 34, 50 etc)",
    ignore=False  # 構造が変われば別実験扱い
  )
  
  pretrained: bool = conf_field(
    default=False,
    desc="ImageNet事前学習済み重みを使用するか",
    ignore=False  # 重みが変われば結果も変わる
  )
  
  lr: float = conf_field(
    default=0.001,
    desc="学習率 (Learning Rate)",
    ignore=False  # ハイパーパラメータ
  )