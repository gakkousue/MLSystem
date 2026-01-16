# definitions/datasets/mnist/config.py
# Dataset固有の設定

from dataclasses import dataclass
from MLsystem.utils.config_base import BaseConfig, conf_field

@dataclass
class DatasetConfig(BaseConfig):
  _name: str = "mnist"

  # env_config.json廃止に伴い、デフォルト値を固定パスに設定
  data_dir: str = conf_field(
    default="./data",
    desc="データの保存先ディレクトリ",
    ui_mode="readonly",
    ignore=True   # 保存場所が変わっても実験結果は変わらない
  )

  val_ratio: float = conf_field(
    default=0.2,
    desc="学習データから切り出す検証データ(Validation)の割合",
    ignore=False  # 分割が変われば結果が変わる
  )

  num_workers: int = conf_field(
    default=2,
    desc="データ読み込みの並列数",
    ignore=True   # 速度設定なので除外
  )