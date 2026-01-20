# definitions/datasets/mnist/loaders/base.py
import os
import pickle
import torch
from torchvision import datasets, transforms

class MNISTLoader:
    def __init__(self, data_dir: str):
        """
        MNISTデータローダーの初期化
        Args:
            data_dir (str): データセットのルートディレクトリ
        """
        self.dataset_name = "mnist"
        # base_dir: data_dir/mnist
        self.base_dir = os.path.join(data_dir, self.dataset_name)
        
        self.cache_dir = os.path.join(self.base_dir, "cache")
        self.raw_dir = os.path.join(self.base_dir, "raw")
        
        # キャッシュファイルのパス
        self.train_all_pkl = os.path.join(self.cache_dir, "train_all.pkl")
        self.test_pkl = os.path.join(self.cache_dir, "test.pkl")

    def prepare(self):
        """
        生データをダウンロードし、pickleキャッシュを作成する。
        すでにキャッシュが存在する場合は何もしない。
        """
        if os.path.exists(self.train_all_pkl) and os.path.exists(self.test_pkl):
            return

        print("[MNISTLoader] キャッシュを作成中...")
        os.makedirs(self.cache_dir, exist_ok=True)

        to_tensor = transforms.ToTensor()
        train_raw = datasets.MNIST(
            root=self.raw_dir, train=True, download=True, transform=to_tensor
        )
        test_raw = datasets.MNIST(
            root=self.raw_dir, train=False, download=True, transform=to_tensor
        )

        def to_uint8_tuple(dataset):
            data_list = []
            target_list = []
            for img, target in dataset:
                # float(0.0-1.0) -> byte(0-255) に変換して軽量化
                img_byte = (img * 255).to(torch.uint8)
                data_list.append(img_byte)
                target_list.append(target)
            return torch.stack(data_list), torch.tensor(target_list)

        with open(self.train_all_pkl, "wb") as f:
            pickle.dump(to_uint8_tuple(train_raw), f)

        with open(self.test_pkl, "wb") as f:
            pickle.dump(to_uint8_tuple(test_raw), f)

        print(f"[MNISTLoader] キャッシュを保存しました: {self.cache_dir}")

    def load_train_all(self):
        """学習用（全量）データをロードして返す"""
        with open(self.train_all_pkl, "rb") as f:
            return pickle.load(f)

    def load_test(self):
        """テストデータをロードして返す"""
        with open(self.test_pkl, "rb") as f:
            return pickle.load(f)
