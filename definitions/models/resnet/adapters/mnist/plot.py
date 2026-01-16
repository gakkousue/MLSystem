# definitions/models/resnet/adapters/mnist/plot.py
import os
import matplotlib.pyplot as plt
import torchvision
from MLsystem.utils.base_plot import BasePlot


class AdapterTransformedSamples(BasePlot):
    """
    Adapterによる変換後のデータ(Tensor)を可視化して保存する。
    モデルに入力される直前のデータを確認できる。
    """

    name = "Adapter Transformed Samples"
    description = "Adapter変換後（モデル入力直前）のデータを表示します"

    def execute(self):
        print(f"[{self.name}] Fetching transformed samples...")

        # Datasetのセットアップ
        _, datamodule = self.loader.setup(stage="fit")

        # 学習用データローダーからデータセットを取得
        # Note: datamodule.train_dataloader() は毎回新しいLoaderを作るため、
        #       ここでは dataset プロパティにアクセスする
        datamodule.setup(stage="fit")
        dataset = datamodule.train_ds

        if len(dataset) == 0:
            print("No data in dataset.")
            return

        # 1サンプル取得 (img_tensor, label)
        img_tensor, label = dataset[0]

        # img_tensor は (C, H, W) の形状で、正規化されている可能性がある
        # バッチ次元がない場合は追加 (C, H, W) -> (1, C, H, W)
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # グリッド画像の作成 (normalize=Trueで視認性を確保)
        grid = torchvision.utils.make_grid(img_tensor, normalize=True)

        plt.figure(figsize=(4, 4))
        # (C, H, W) -> (H, W, C)
        plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
        plt.title(f"Transformed Sample (Label: {label})")
        plt.axis("off")

        save_path = os.path.join(self.loader.exp_dir, "adapter_transformed.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[{self.name}] Saved: {save_path}")
