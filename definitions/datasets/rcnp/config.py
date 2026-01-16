# definitions/datasets/rcnp/config.py
from dataclasses import dataclass
from MLsystem.utils.config_base import BaseConfig, conf_field


@dataclass
class DatasetConfig(BaseConfig):
    _name: str = "rcnp"

    data_dir: str = conf_field(
        default="./data/h5_files",
        desc="データファイル(.h5)が保存されているディレクトリ",
        ignore=True,
    )

    max_particles: int = conf_field(
        default=100,
        desc="1ジェットあたりの最大粒子数（パディング/切り捨て用）",
        ignore=False,
    )

    train_ratio: float = conf_field(default=0.6, desc="学習データの割合", ignore=False)

    val_ratio: float = conf_field(
        default=0.2, desc="検証データの割合 (残りがテストデータ)", ignore=False
    )

    num_workers: int = conf_field(default=2, desc="データ読み込みの並列数", ignore=True)
