# definitions/datasets/mnist/plots.py
import matplotlib.pyplot as plt
import torchvision
import os
import torch
from torchvision import transforms

def plot_samples(datamodule, output_dir):
    """
    データセットの生サンプル(Adapter適用前)をグリッドで保存する。
    Adapterの影響を受ける前の、人間が見やすい状態を確認するためのもの。
    """
    print("Generating raw data sample plot...")
    
    # データの準備 (setupを呼んでデータをロードさせる)
    # ここでは生データを見るため、あえて transform を無効化した一時的なDatasetを作るか、
    # datamodule内部のデータに直接アクセスする
    
    # setupがまだなら呼ぶ
    if not hasattr(datamodule, "train_all_pkl"):
        # パス設定のため__init__相当が必要だが、インスタンスは渡されている前提
        pass
        
    # 生データをロード (setup依存を避けるため直接ロードを試みる)
    pkl_path = datamodule.train_all_pkl
    if not os.path.exists(pkl_path):
        datamodule.prepare_data()
    
    import pickle
    with open(pkl_path, "rb") as f:
        data_tensor, _ = pickle.load(f)
        
    # 先頭16枚を取得
    # data_tensor は ByteTensor (N, C, H, W) 
    images = data_tensor[:16]
    
    # float(0-1)に戻す
    images = images.float() / 255.0
    
    # グリッド画像の作成
    grid = torchvision.utils.make_grid(images, nrow=4, normalize=False)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
    plt.title("MNIST Raw Samples (Before Adapter)")
    plt.axis('off')
    
    save_path = os.path.join(output_dir, "data_raw_samples.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")