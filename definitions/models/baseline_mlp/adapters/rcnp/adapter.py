# definitions/models/baseline_mlp/adapters/rcnp/adapter.py

def get_input_transform(adapter_conf):
    """
    RCNPデータセットは構造化データであり、DataModule内でTensor化されるため、
    Adapterとしての追加変換は不要。
    """
    return None

def get_model_init_args(data_meta, adapter_conf):
    """
    DataModuleのメタ情報をModelの引数に変換する。
    """
    num_classes = data_meta.get("num_classes", 3)
    
    return {
        "num_classes": num_classes
    }