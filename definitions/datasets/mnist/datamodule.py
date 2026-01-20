# definitions/datasets/mnist/datamodule.py
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

class BaseDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        """
        メモリ上のTensorデータをDatasetとしてラップする
        Args:
            data (Tensor): 画像データ (N, C, H, W) or (N, H, W)
            targets (Tensor): ラベルデータ
            transform (callable, optional): 適用する変換
        """
        self.data = data
        self.targets = targets
        self.transform = transform
        self.to_pil = transforms.ToPILImage()

    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]

        # ByteTensor -> PIL Image に戻してTransformを適用可能にする
        if isinstance(img, torch.Tensor):
            img = self.to_pil(img)

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


class DataModule(pl.LightningDataModule):
    def __init__(self, loader, batch_size=32, num_workers=0, val_ratio=0.2, seed=42):
        """
        Args:
            loader: データの読み込みを担当するLoaderインスタンス (Hydraで注入)
            batch_size (int): バッチサイズ
            num_workers (int): DataLoaderのワーカー数
            val_ratio (float): 検証データの割合
            seed (int): ランダムシード
        """
        super().__init__()
        self.loader = loader
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.seed = seed
        
        # 内部状態
        self.train_ds = None
        self.val_ds = None

    def prepare_data(self):
        """データのダウンロードとキャッシュ作成（Loaderに委譲）"""
        self.loader.prepare()

    def setup(self, stage=None):
        """
        データをロードし、学習/検証セットに分割する
        """
        if stage == "fit" or stage is None:
            # Loaderから全データを取得
            all_data, all_targets = self.loader.load_train_all()

            # データセット分割
            total_len = len(all_data)
            val_len = int(total_len * self.val_ratio)
            train_len = total_len - val_len

            # Transformはここでは基本的なToTensorのみ適用し、
            # モデル固有の正規化などはModel内のTransformまたはここで追加引数として受け取る設計が望ましいが、
            # シンプルにするため一旦標準的なToTensorとする。
            # 必要であればHydra設定からtransformsを注入することも可能。
            common_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

            full_ds = BaseDataset(all_data, all_targets, transform=common_transform)

            # 再現性のあるランダム分割
            gen = torch.Generator().manual_seed(self.seed)
            self.train_ds, self.val_ds = random_split(
                full_ds, [train_len, val_len], generator=gen
            )

            print(f"[DataModule] Dataset split: Train={len(self.train_ds)}, Val={len(self.val_ds)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
