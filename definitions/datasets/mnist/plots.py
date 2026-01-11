# definitions/datasets/mnist/plots.py
import matplotlib.pyplot as plt
import torchvision
import os
import pickle
from system.utils.base_plot import BasePlot

class RawDataSamples(BasePlot):
    """
    データセットの生サンプル(Adapter適用前)をグリッドで保存する。
    Adapterの影響を受ける前の、人間が見やすい状態を確認するためのもの。
    """
    name = "Raw Data Samples"
    description = "Adapter適用前の生データを表示します"

    def execute(self):
        print(f"[{self.name}] Generating raw data sample plot...")
        
        # Datamoduleのセットアップ (データパス等の取得のため)
        _, datamodule = self.loader.setup(stage="fit")
        
        # 生データをロード (prepare_dataが呼ばれている前提)
        # Loader経由ならsetup内でprepare_dataも呼ばれている
        pkl_path = datamodule.train_all_pkl
        if not os.path.exists(pkl_path):
            datamodule.prepare_data()
            
        with open(pkl_path, "rb") as f:
            data_tensor, _ = pickle.load(f)
            
        # 先頭16枚を取得 (ByteTensor: N, C, H, W)
        images = data_tensor[:16]
        
        # float(0-1)に戻す
        images = images.float() / 255.0
        
        # グリッド画像の作成
        grid = torchvision.utils.make_grid(images, nrow=4, normalize=False)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
        plt.title("MNIST Raw Samples (Before Adapter)")
        plt.axis('off')
        
        save_path = os.path.join(self.loader.exp_dir, "data_raw_samples.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[{self.name}] Saved: {save_path}")