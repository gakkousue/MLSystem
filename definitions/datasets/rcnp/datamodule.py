# definitions/datasets/rcnp/datamodule.py
import os
import glob
import re
import pickle
import random
import numpy as np
import pandas as pd
import h5py
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class RCNPDataset(Dataset):
    # transform引数を追加して保存する
    def __init__(self, data_dict, indices, transform=None):
        """
        data_dict: 全データの辞書
        indices: このデータセットで使用するインデックスのリスト
        """
        self.axis = torch.from_numpy(data_dict["axis"][indices])
        self.part = torch.from_numpy(data_dict["part"][indices])
        self.num = torch.from_numpy(data_dict["num"][indices]).float()
        self.label = torch.from_numpy(data_dict["label"][indices]).long()
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        # データを取得
        a = self.axis[i]
        p = self.part[i]
        n = self.num[i]
        l = self.label[i]

        # Adapterで定義された正規化処理があれば適用する
        # transformは (axis, part) を受け取って変換後の (axis, part) を返す関数を想定
        if self.transform:
            a, p = self.transform(a, p)

        return a, p, n, l


class DataModule(pl.LightningDataModule):
    def __init__(self, adapter_transform=None, **kwargs):
        super().__init__()
        self.conf = kwargs
        # 受け取ったtransform関数をメンバ変数として保存する
        self.adapter_transform = adapter_transform

        self.data_dir = self.conf["data_dir"]
        self.max_particles = self.conf.get("max_particles", 100)

        # クラス定義 (bb=0, cc=1, uds=2)
        self.class_names = ["bb", "cc", "uds"]
        self.num_classes = 3

        self.input_dims = {"axis": 3, "part": 7}

    def _pad_particles(self, arrays_list):
        """
        可変長の粒子特徴量配列のリストを、(N, MaxPart, Features) にパディングする。
        arrays_list: List of (n_particles, n_features) arrays
        """
        batch_size = len(arrays_list)
        n_feats = 7

        # 出力バッファ確保
        padded = np.zeros((batch_size, self.max_particles, n_feats), dtype=np.float32)

        for i, arr in enumerate(arrays_list):
            # 長さチェック
            length = min(len(arr), self.max_particles)
            if length > 0:
                # pandasのセルに保存されたarrayが(n_part,)などの場合があるためshape確認
                # 2次元配列 (n_part, n_feats) を想定
                if hasattr(arr, "shape") and len(arr.shape) == 2:
                    padded[i, :length, :] = arr[:length]
                else:
                    # まれにリスト等で入っている場合のケア
                    arr_np = np.array(arr)
                    if len(arr_np.shape) == 2:
                        padded[i, :length, :] = arr_np[:length]

        return padded

    def prepare_data(self):
        """ディレクトリの存在確認のみ"""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def _load_raw_data(self):
        """H5ファイルを読み込み、指定された固定イベント数でデータを整形して返す"""
        print(f">> Scanning H5 files in {self.data_dir} ...")

        # 固定のイベント数を定義
        load_counts = {
            "bb": 60000, "cc": 60000, "uu": 40000, "dd": 10000, "ss": 10000,
        }
        label_map = {"bb": 0, "cc": 1, "uu": 2, "dd": 2, "ss": 2}
        tags = list(load_counts.keys())
        
        # 1ファイルあたりの想定イベント数
        EVENTS_PER_FILE = 50000

        # ファイルを分類して収集: map[tag][pol] = [files...]
        files_map = {tag: {"eL": [], "eR": []} for tag in tags}
        
        all_files = glob.glob(os.path.join(self.data_dir, "*.h5"))
        if not all_files:
            raise RuntimeError(f"No .h5 files found in {self.data_dir}")

        for fpath in all_files:
            fname = os.path.basename(fpath)
            # フレーバー判定 (先頭2文字)
            m_tag = re.match(r"^([a-z]{2})", fname)
            if not m_tag or m_tag.group(1) not in tags:
                continue
            tag = m_tag.group(1)

            # 偏極判定 (ファイル名に eL または eR が含まれると仮定)
            if "eL" in fname:
                pol = "eL"
            elif "eR" in fname:
                pol = "eR"
            else:
                continue # 偏極が特定できないファイルはスキップ

            files_map[tag][pol].append(fpath)

        # 必要なイベント数を供給できるかチェック
        print(">> Checking file availability for fixed event counts:")
        for tag, total_needed in load_counts.items():
            for pol in ["eL", "eR"]:
                half_needed = total_needed // 2
                available_files = len(files_map[tag][pol])
                available_events = available_files * EVENTS_PER_FILE
                print(f"  [{tag}][{pol}] Needed: {half_needed}, Available: {available_events} ({available_files} files)")
                if available_events < half_needed:
                    raise RuntimeError(f"Insufficient data for [{tag}][{pol}]. Needed {half_needed} events, but only {available_events} available.")

        # データのロード
        all_axis = []
        all_part_lists = []
        all_nums = []
        all_labels = []

        cols_axis = ["jet_px", "jet_py", "jet_pz"]
        # 荷電粒子と中性粒子の特徴量カラムを定義
        cols_part_charged = [
            "pfcand_px", "pfcand_py", "pfcand_pz", "pfcand_e",
            "pfcand_charge", "pfcand_dxy", "pfcand_dz",
        ]
        cols_part_neutral = [
            "neu_pfcand_px", "neu_pfcand_py", "neu_pfcand_pz", "neu_pfcand_e",
            "neu_pfcand_charge", "neu_pfcand_dxy", "neu_pfcand_dz",
        ]

        for tag, total_needed in load_counts.items():
            label = label_map[tag]
            # eL, eR から半分ずつ読み込む
            half_needed = total_needed // 2
            
            for pol in ["eL", "eR"]:
                needed = half_needed
                # ファイルリストをソートして順に読み込む
                file_list = sorted(files_map[tag][pol])
                
                for fpath in file_list:
                    if needed <= 0:
                        break
                    
                    fname = os.path.basename(fpath)
                    try:
                        with h5py.File(fpath, "r") as f:
                            grp = f["ntp"]
                            
                            # Axis
                            axis_cols_data = [grp[c][:] for c in cols_axis]
                            file_axis = np.stack(axis_cols_data, axis=1).astype(np.float32)
                            n_events_in_file = len(file_axis)

                            # 今回このファイルから取得する数
                            take_n = min(needed, n_events_in_file)
                            
                            # データをスライスして取得
                            sliced_axis = file_axis[:take_n]
                            all_axis.append(sliced_axis)
                            
                            all_labels.append(np.full(take_n, label, dtype=np.int64))

                            # Particles (荷電粒子と中性粒子を結合)
                            charged_part_dict = {c: grp[c][:take_n] for c in cols_part_charged}
                            neutral_part_dict = {c: grp[c][:take_n] for c in cols_part_neutral}

                            for i in range(take_n):
                                # 荷電粒子の特徴量を取得
                                feats_charged = [charged_part_dict[c][i] for c in cols_part_charged]
                                if len(feats_charged) > 0 and len(feats_charged[0]) > 0:
                                    stacked_charged = np.stack(feats_charged, axis=1).astype(np.float32)
                                else:
                                    stacked_charged = np.zeros((0, 7), dtype=np.float32)

                                # 中性粒子の特徴量を取得
                                feats_neutral = [neutral_part_dict[c][i] for c in cols_part_neutral]
                                if len(feats_neutral) > 0 and len(feats_neutral[0]) > 0:
                                    stacked_neutral = np.stack(feats_neutral, axis=1).astype(np.float32)
                                else:
                                    stacked_neutral = np.zeros((0, 7), dtype=np.float32)

                                # 荷電粒子と中性粒子を結合
                                stacked = np.vstack([stacked_charged, stacked_neutral])

                                all_part_lists.append(stacked)
                                all_nums.append(len(stacked))
                            
                            needed -= take_n

                    except Exception as e:
                        print(f"Error reading {fname}: {e}")

        if not all_axis:
            raise RuntimeError("No valid data loaded.")

        full_axis = np.concatenate(all_axis, axis=0)
        full_labels = np.concatenate(all_labels, axis=0)
        full_nums = np.array(all_nums, dtype=np.float32)
        full_part = self._pad_particles(all_part_lists)

        return {
            "axis": full_axis,
            "part": full_part,
            "num": full_nums,
            "label": full_labels,
        }

    def setup(self, stage=None):
        data_dict = self._load_raw_data()

        total_len = len(data_dict["label"])
        indices = list(range(total_len))

        # Shuffle
        rng = random.Random(self.conf.get("seed", 42))
        rng.shuffle(indices)

        # Split
        train_ratio = self.conf["train_ratio"]
        val_ratio = self.conf["val_ratio"]

        s1 = int(total_len * train_ratio)
        s2 = int(total_len * (train_ratio + val_ratio))

        ind_train = indices[:s1]
        ind_val = indices[s1:s2]
        ind_test = indices[s2:]

        # transform (self.adapter_transform) を渡すように修正
        self.train_ds = RCNPDataset(
            data_dict, ind_train, transform=self.adapter_transform
        )
        self.val_ds = RCNPDataset(data_dict, ind_val, transform=self.adapter_transform)
        self.test_ds = RCNPDataset(
            data_dict, ind_test, transform=self.adapter_transform
        )

        print(
            f"Dataset Split: Train={len(self.train_ds)}, Val={len(self.val_ds)}, Test={len(self.test_ds)}"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.conf["batch_size"],
            num_workers=self.conf["num_workers"],
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.conf["batch_size"],
            num_workers=self.conf["num_workers"],
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.conf["batch_size"],
            num_workers=self.conf["num_workers"],
            shuffle=False,
        )
