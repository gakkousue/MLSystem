# system/hashing.py
import hashlib
import json
import copy

def compute_combined_hash(common_schema, common_params, 
                          model_schema, model_params, 
                          adapter_schema, adapter_params,
                          data_schema, data_params):
    """
    パラメータからハッシュIDを生成する。
    スキーマで "ignore": True とされている項目は、値が何であれハッシュ計算から除外する。
    """
    payload = {
        "model": model_params.get("_name"),
        "adapter": adapter_params.get("_name"),
        "dataset": data_params.get("_name"),
        "common_diff": {},
        "model_diff": {},
        "adapter_diff": {},
        "data_diff": {}
    }

    def process_params(schema, params, target_dict):
        for k, info in schema.items():
            # ignoreフラグがTrueなら、値がどうあれハッシュ計算には含めない
            if info.get("ignore", False):
                continue
            
            default = info["default"]
            val = params.get(k, default)
            
            # デフォルト値と異なる場合のみ記録
            if val != default:
                target_dict[k] = val

    # 各カテゴリの差分抽出
    process_params(common_schema, common_params, payload["common_diff"])
    process_params(model_schema, model_params, payload["model_diff"])
    process_params(adapter_schema, adapter_params, payload["adapter_diff"]) # 追加
    process_params(data_schema, data_params, payload["data_diff"])

    # JSON化してハッシュ計算
    encoded = json.dumps(payload, sort_keys=True).encode('utf-8')
    hash_id = hashlib.md5(encoded).hexdigest()[:10]

    return hash_id, payload