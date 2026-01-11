# definitions/datasets/mnist/datamodule.py
import os
import pickle
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
# CONFIG_SCHEMA のインポートを削除

# Adapterの変換を適用するための独自Datasetクラス
class BaseDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        # MNISTの生データはTensor(Byte)またはPILを想定
        # AdapterのToTensorはPILまたはndarrayを期待するため、変換用のPIL化ツールを用意
        self.to_pil = transforms.ToPILImage()

    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]
        
        # 保存形式がTensor(Byte)の場合、PILに戻してからAdapterのTransform(ToTensor等)に通す
        # これによりAdapter側は一般的な "PIL -> Tensor" のTransformを書くだけで済む
        if isinstance(img, torch.Tensor):
            img = self.to_pil(img)
            
        if self.transform:
            img = self.transform(img)
            
        return img, target

    def __len__(self):
        return len(self.data)

class DataModule(pl.LightningDataModule):
    # 初期化時にAdapterからの変換関数(adapter_transform)を受け取る
    def __init__(self, adapter_transform=None, **kwargs):
        super().__init__()
        
        # kwargsをそのまま設定として使用する
        self.conf = kwargs
        self.adapter_transform = adapter_transform
            
        # メタ情報
        self.num_classes = 10
            
        # メタ情報
        self.num_classes = 10
        self.num_channels = 1
        
        # パス定義
        dataset_name = "mnist"
        base_dir = os.path.join(self.conf["data_dir"], dataset_name)
        
        self.cache_dir = os.path.join(base_dir, "cache")
        self.raw_dir = os.path.join(base_dir, "raw")
        
        # キャッシュファイル: 全学習データとテストデータ
        self.train_all_pkl = os.path.join(self.cache_dir, "train_all.pkl")
        self.test_pkl = os.path.join(self.cache_dir, "test.pkl")

    def prepare_data(self):
        """
        生データをダウンロードし、Adapterの影響を受けない形式(ByteTensor)でキャッシュする
        """
        if os.path.exists(self.train_all_pkl) and os.path.exists(self.test_pkl):
            return

        print("Creating pickle cache for MNIST...")
        os.makedirs(self.cache_dir, exist_ok=True)

        # 生データの取得 (transform=ToTensor を指定して一旦Tensorにする)
        # MNISTの生画像はPILだが、保存効率のためByteTensor(0-255)にする
        to_tensor = transforms.ToTensor()
        train_raw = datasets.MNIST(root=self.raw_dir, train=True, download=True, transform=to_tensor)
        test_raw = datasets.MNIST(root=self.raw_dir, train=False, download=True, transform=to_tensor)

        def to_uint8_tuple(dataset):
            data_list = []
            target_list = []
            for img, target in dataset:
                # float(0.0-1.0) -> byte(0-255)
                img_byte = (img * 255).to(torch.uint8)
                data_list.append(img_byte)
                target_list.append(target)
            return torch.stack(data_list), torch.tensor(target_list)

        # 保存
        with open(self.train_all_pkl, "wb") as f:
            pickle.dump(to_uint8_tuple(train_raw), f)
            
        with open(self.test_pkl, "wb") as f:
            pickle.dump(to_uint8_tuple(test_raw), f)
            
        print(f"Pickle cache saved to {self.cache_dir}")

    def setup(self, stage=None):
        """
        キャッシュをロードし、学習/検証に分割してDatasetを作成する
        """
        if stage == "fit" or stage is None:
            # 学習データをロード
            with open(self.train_all_pkl, "rb") as f:
                all_data, all_targets = pickle.load(f)
            
            # 分割計算
            total_len = len(all_data)
            val_len = int(total_len * self.conf["val_ratio"])
            train_len = total_len - val_len
            
            # 再現性のある分割
            # データセット全体をラップしてからsplitする
            full_ds = BaseDataset(all_data, all_targets, transform=self.adapter_transform)
            
            # torch.utils.data.random_split を使用 (seedはGlobal設定に依存)
            # Generatorを指定して再現性を担保
            gen = torch.Generator().manual_seed(self.conf.get("seed", 42))
            self.train_ds, self.val_ds = random_split(full_ds, [train_len, val_len], generator=gen)

            print(f"Dataset split: Train={len(self.train_ds)}, Val={len(self.val_ds)}")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.conf["batch_size"], 
                          num_workers=self.conf["num_workers"], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.conf["batch_size"], 
                          num_workers=self.conf["num_workers"], shuffle=False)

def create_datamodule(conf, adapter_transform=None):
    return DataModule(adapter_transform=adapter_transform, **conf)