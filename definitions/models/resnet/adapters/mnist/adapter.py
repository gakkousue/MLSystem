# definitions/models/resnet/adapters/mnist/adapter.py
import torch.nn as nn
from torchvision import transforms

def get_input_transform(adapter_conf):
    """
    データセット(Dataset)に適用させる変換関数を返す。
    ここで返す変換は、Dataset側でのAugmentationの「後」に適用される。
    """
    transform_list = []
    
    # リサイズ (もし設定で倍率が変えられていれば)
    scale = adapter_conf.get("resize_scale", 1.0)
    if scale != 1.0:
        # MNISTは28x28なので、そこからの倍率で計算
        new_size = int(28 * scale)
        transform_list.append(transforms.Resize((new_size, new_size)))
        
    # 必須変換: Tensor化
    transform_list.append(transforms.ToTensor())
    
    # 必須変換: 正規化 (MNISTの平均・分散)
    # これにより入力は 0~1 ではなく、負の値も含む分布になる
    transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    
    return transforms.Compose(transform_list)

def get_model_init_args(data_meta, adapter_conf):
    """
    DataModuleのメタ情報を受け取り、Modelの__init__に渡す引数辞書を生成する。
    Adapterは「モデルがどんな引数を期待しているか」を知っている。
    """
    # Datasetから得られたチャネル数とクラス数
    # もし情報がなければデフォルト値を使用
    in_channels = data_meta.get("num_channels", 1)
    num_classes = data_meta.get("num_classes", 10)
    
    return {
        "in_channels": in_channels,
        "num_classes": num_classes
    }