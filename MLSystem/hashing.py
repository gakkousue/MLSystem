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
    # ハッシュ計算用の別途用意
    hash_payload = {
        "model": payload["model"],
        "adapter": payload["adapter"],
        "dataset": payload["dataset"],
        "common_diff": {},
        "model_diff": {},
        "adapter_diff": {},
        "data_diff": {},
    }

    def process_params(config_cls, params, target_diff, hash_diff):
        # config_cls が None の場合はスキップ
        if config_cls is None:
            return

        for f in fields(config_cls):
            default = f.default
            val = params.get(f.name, default)

            # excludes処理: 特定のキーをハッシュ計算から除外
            excludes = f.metadata.get("excludes")
            if excludes and isinstance(val, dict):
                val = copy.deepcopy(val)
                for ex_key in excludes:
                    if ex_key in val:
                        del val[ex_key]
            
            # デフォルト値と異なる場合のみ記録
            if val != default:
                target_diff[f.name] = val
                # ignoreフラグがFalse（または未設定）ならハッシュ計算対象に含める
                if not f.metadata.get("ignore", False):
                    hash_diff[f.name] = val

    # 各カテゴリの差分抽出
    process_params(common_cls, common_params, payload["common_diff"], hash_payload["common_diff"])
    process_params(model_cls, model_params, payload["model_diff"], hash_payload["model_diff"])
    process_params(adapter_cls, adapter_params, payload["adapter_diff"], hash_payload["adapter_diff"])
    process_params(data_cls, data_params, payload["data_diff"], hash_payload["data_diff"])

    # ハッシュ計算用ペイロードをJSON化
    encoded = json.dumps(hash_payload, sort_keys=True).encode("utf-8")
    hash_id = hashlib.md5(encoded).hexdigest()[:10]

    return hash_id, payload
