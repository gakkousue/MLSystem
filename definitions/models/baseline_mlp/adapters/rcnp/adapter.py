# definitions/models/baseline_mlp/adapters/rcnp/adapter.py

def get_input_transform(adapter_conf):
    """
    RCNPデータセットは値が大きいため、ニューラルネットワークに入力する前に
    適切な範囲にスケーリング（正規化）を行う必要がある。
    """
    class RCNPScaler:
        def __init__(self, scale=100.0):
            self.scale = scale
            
        def __call__(self, axis, part):
            # 単純に定数で割って値を小さくする
            # データ分布に応じて調整が必要だが、まずは発散を防ぐために1/100程度にする
            return axis / self.scale, part / self.scale

    return RCNPScaler(scale=100.0)

def get_model_init_args(data_meta, adapter_conf):
    """
    DataModuleのメタ情報をModelの引数に変換する。
    """
    num_classes = data_meta.get("num_classes", 3)
    
    return {
        "num_classes": num_classes
    }