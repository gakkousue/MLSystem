# system/utils/config_base.py
from dataclasses import dataclass, field
from typing import Any

def conf_field(default: Any, desc: str = "", ui_mode: str = "input", ignore: bool = False, **kwargs):
  """
  GUI用メタデータを含むdataclassフィールドを生成するヘルパー関数
  
  Args:
    default: デフォルト値
    desc: 設定の説明文
    ui_mode: "input"(通常), "readonly"(読取専用), "hidden"(非表示)
    ignore: Trueの場合、ハッシュ計算（実験ID）の算出元から除外する
  """
  metadata = {
    "desc": desc,
    "ui_mode": ui_mode,
    "ignore": ignore,
  }
  metadata.update(kwargs)
  return field(default=default, metadata=metadata)

@dataclass
class BaseConfig:
  """すべての設定クラスの基底"""
  # 内部識別用の名前。ハッシュ計算には含めるが、GUI等では隠蔽する。
  _name: str = field(default="base", metadata={"ui_mode": "hidden"})