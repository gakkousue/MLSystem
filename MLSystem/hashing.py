# system/hashing.py
import hashlib
import json
import copy

from dataclasses import fields


def compute_combined_hash(
    common_cls,
    common_params,
    model_cls,
    model_params,
    adapter_cls,
    adapter_params,
    data_cls,
    data_params,
):
    """
    Configクラス定義とパラメータからハッシュIDを生成する。
    dataclassのmetadataにある "ignore": True の項目は除外する。
    """
    payload = {
        "model": model_params.get("_name"),
        "adapter": adapter_params.get("_name"),
        "dataset": data_params.get("_name"),
        "common_diff": {},
        "model_diff": {},
        "adapter_diff": {},
        "data_diff": {},
    }

    def process_params(config_cls, params, target_dict):
        # config_cls が None の場合はスキップ
        if config_cls is None:
            return

        for f in fields(config_cls):
            # ignoreフラグがTrueなら除外
            if f.metadata.get("ignore", False):
                continue

            # 内部管理用フィールド(_nameなど)も除外したければここで判定
            # ただし _name は ignore=True になっていなくても値が変わらないはずなので問題ない

            default = f.default
            val = params.get(f.name, default)

            # デフォルト値と異なる場合のみ記録
            # 型が違う場合(int vs float)の比較には注意が必要だが、
            # 基本的にGUI/CLIからの入力値とデフォルト値の比較を行う
            if val != default:
                target_dict[f.name] = val

    # 各カテゴリの差分抽出
    process_params(common_cls, common_params, payload["common_diff"])
    process_params(model_cls, model_params, payload["model_diff"])
    process_params(adapter_cls, adapter_params, payload["adapter_diff"])
    process_params(data_cls, data_params, payload["data_diff"])

    # JSON化してハッシュ計算
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    hash_id = hashlib.md5(encoded).hexdigest()[:10]

    return hash_id, payload
