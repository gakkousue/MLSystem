# definitions/models/resnet/adapters/mnist/config.py
# ResNet用 MNISTアダプター設定

from dataclasses import dataclass
from MLsystem.utils.config_base import BaseConfig, conf_field


@dataclass
class AdapterConfig(BaseConfig):
    _name: str = "mnist"

    target_dataset: str = conf_field(
        default="mnist",
        desc="対象データセット (固定)",
        ui_mode="hidden",
        ignore=True,  # Dataset設定側で管理されるため除外
    )

    resize_scale: float = conf_field(
        default=1.0,
        desc="画像のリサイズ倍率 (1.0 = そのまま)",
        ignore=False,  # 入力データが変わるのでハッシュに含める
    )
