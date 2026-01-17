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

            default = f.default
            val = params.get(f.name, default)

            # excludes処理: 特定のキーをハッシュ計算から除外
            excludes = f.metadata.get("excludes")
            if excludes and isinstance(val, dict):
                # 元のdictを破壊しないようコピー
                val = copy.deepcopy(val)
                for ex_key in excludes:
                    if ex_key in val:
                        del val[ex_key]
            
            # デフォルト値と異なる場合のみ記録
            # 注意: excludes適用後の値をデフォルトと比較するのは難しい（デフォルト値にもexcludes適用が必要？）
            # ここでは「excludes適用後のval」をハッシュ計算に使う。
            # default判定は「元の値」で行うべきか、「適用後の値」で行うべきか。
            # 通常、default値自体もハッシュに影響しない（省略時と同じ）ため、
            # val != default のチェックは「入力された値がデフォルトと違うか」を見るもの。
            # excludesは「その値の中身の一部を無視する」もの。
            # シンプルに、「値が存在すれば(デフォルトと違えば)登録し、その中身を加工する」のが安全。
            
            if val != default:
                 # さらに、valが辞書の場合、除外後の辞書が空になったら記録すべきか？
                 # 空になっても「空の辞書を設定した」という意味があるなら記録すべき。
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
