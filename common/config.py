# common/config.py
# 全モデル共通の設定 (max_epochs, batch_size等)

from dataclasses import dataclass
from system.utils.config_base import BaseConfig, conf_field

@dataclass
class CommonConfig(BaseConfig):
  _name: str = "common"

  max_epochs: int = conf_field(
    default=10,
    desc="最大エポック数",
    ignore=True   # ハッシュ計算から除外（途中再開可能にするため）
  )
  
  batch_size: int = conf_field(
    default=32,
    desc="バッチサイズ",
    ignore=False  # 結果に影響するためハッシュに含める
  )
  
  seed: int = conf_field(
    default=42,
    desc="乱数シード",
    ignore=False  # 再現性のためハッシュに含める
  )
  
  num_workers: int = conf_field(
    default=2,
    desc="データ読み込み並列数",
    ignore=True   # 計算速度のみの影響なのでハッシュから除外
  )
  
  gpus: int = conf_field(
    default=1,
    desc="使用GPU数 (0=CPU)",
    ignore=True   # インフラ設定なので除外
  )