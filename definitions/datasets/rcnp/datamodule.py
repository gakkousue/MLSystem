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
    def __init__(self, data_dict, indices):
        """
        data_dict: キャッシュからロードされた全データの辞書
          - 'axis': (N, 3) numpy array
          - 'part': (N, MaxPart, 7) numpy array (padded)
          - 'num': (N,) numpy array
          - 'label': (N,) numpy array
        indices: このデータセットで使用するインデックスのリスト
        """
        self.axis = torch.from_numpy(data_dict['axis'][indices])
        self.part = torch.from_numpy(data_dict['part'][indices])
        self.num  = torch.from_numpy(data_dict['num'][indices]).float()
        self.label = torch.from_numpy(data_dict['label'][indices]).long()

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, i):
        return self.axis[i], self.part[i], self.num[i], self.label[i]


class DataModule(pl.LightningDataModule):
    def __init__(self, adapter_transform=None, **kwargs):
        super().__init__()
        self.conf = kwargs
        
        self.data_dir = self.conf["data_dir"]
        self.cache_dir = os.path.join(self.data_dir, "cache")
        self.max_particles = self.conf.get("max_particles", 100)
        
        # クラス定義 (bb=0, cc=1, uds=2)
        self.class_names = ["bb", "cc", "uds"]
        self.num_classes = 3
        
        self.input_dims = {
            "axis": 3, 
            "part": 7
        }

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
        """H5ファイルを読み込み、結合・整形してキャッシュを作成"""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
            
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, f"data_cache_len{self.max_particles}.pkl")
        
        if os.path.exists(cache_path):
            return

        print(f">> Scanning H5 files in {self.data_dir} ...")
        
        # ファイルパターンの定義
        # ファイル名例: bb.eL.pR_50000.h5
        # ラベルマッピング: bb->0, cc->1, uu/dd/ss->2
        label_map = {
            "bb": 0, "cc": 1, 
            "uu": 2, "dd": 2, "ss": 2
        }
        
        files = glob.glob(os.path.join(self.data_dir, "*.h5"))
        if not files:
            raise RuntimeError(f"No .h5 files found in {self.data_dir}")
            
        all_axis = []
        all_part_lists = [] # パディング前の一時リスト
        all_nums = []
        all_labels = []
        
        # 特徴量カラム定義
        # TTreeの変数名に基づく
        cols_axis = ["jet_px", "jet_py", "jet_pz"]
        cols_part = [
            "pfcand_px", "pfcand_py", "pfcand_pz", "pfcand_e", 
            "pfcand_charge", "pfcand_dxy", "pfcand_dz"
        ]
        
        for fpath in files:
            fname = os.path.basename(fpath)
            
            # ラベルの特定
            # 先頭のドットまでを取得 (bb, cc, uu...)
            m = re.match(r"^([a-z]{2})", fname)
            if not m:
                print(f"Skipping unknown file format: {fname}")
                continue
                
            tag = m.group(1)
            if tag not in label_map:
                print(f"Skipping unknown tag '{tag}': {fname}")
                continue
                
            label = label_map[tag]
            print(f"Processing {fname} (Label: {tag}->{label})")
            
            # h5pyで読み込み
            try:
                with h5py.File(fpath, 'r') as f:
                    if 'ntp' not in f:
                        print(f"Group 'ntp' not found in {fname}. Keys: {list(f.keys())}")
                        continue
                    
                    grp = f['ntp']
                    
                    # --- Axis Features ---
                    # 各カラムを読み込んで結合 (N, 3)
                    # jet_px, jet_py, jet_pz はそれぞれ (N,) の配列想定
                    axis_cols_data = []
                    for c in cols_axis:
                        if c in grp:
                            axis_cols_data.append(grp[c][:])
                        else:
                            raise KeyError(f"Column '{c}' not found in {fname}")
                    
                    # スタックして (N, 3) に変換
                    axis_data = np.stack(axis_cols_data, axis=1).astype(np.float32)
                    all_axis.append(axis_data)
                    
                    # --- Labels ---
                    n_events = len(axis_data)
                    labels = np.full(n_events, label, dtype=np.int64)
                    all_labels.append(labels)
                    
                    # --- Particle Features ---
                    # pfcand_* は VLEN (Variable Length) データセットとして保存されている想定
                    # h5pyで読むと、各要素が numpy array であるような numpy object array になる (N,)
                    part_dict = {}
                    for c in cols_part:
                        if c in grp:
                            part_dict[c] = grp[c][:]
                        else:
                            raise KeyError(f"Column '{c}' not found in {fname}")
                    
                    # メモリ上の辞書データを使って後続処理へ
                    # part_dict[c] は (N,) の numpy array (object)
                    
                    current_part_list = []
                    current_nums = []
                    
                    for i in range(n_events):
                        # 各特徴量の配列を取得
                        feats = [part_dict[c][i] for c in cols_part]
                        
                        # 転置して (N_part, 7) にする
                        if len(feats[0]) == 0:
                            stacked = np.zeros((0, 7), dtype=np.float32)
                        else:
                            stacked = np.stack(feats, axis=1).astype(np.float32)
                        
                        current_part_list.append(stacked)
                        current_nums.append(len(stacked))
                        
                    all_part_lists.extend(current_part_list)
                    all_nums.extend(current_nums)

            except Exception as e:
                print(f"Failed to read {fname}: {e}")
                continue
                # 各特徴量の配列を取得
                # 例: px = [0.1, 0.2], py = [0.3, 0.4] ...
                feats = [part_dict[c][i] for c in cols_part]
                
                # 転置して (N_part, 7) にする
                # np.stack(feats, axis=1)
                # ただし要素数が0の場合がある
                if len(feats[0]) == 0:
                    stacked = np.zeros((0, 7), dtype=np.float32)
                else:
                    stacked = np.stack(feats, axis=1).astype(np.float32)
                
                current_part_list.append(stacked)
                current_nums.append(len(stacked))
                
            all_part_lists.extend(current_part_list)
            all_nums.extend(current_nums)

        # 結合
        if not all_axis:
            raise RuntimeError("No valid data loaded.")

        full_axis = np.concatenate(all_axis, axis=0)
        full_labels = np.concatenate(all_labels, axis=0)
        full_nums = np.array(all_nums, dtype=np.float32)
        
        print(f">> Padding particle features (Max={self.max_particles})...")
        full_part = self._pad_particles(all_part_lists)
        
        # 保存
        data_dict = {
            "axis": full_axis,
            "part": full_part,
            "num": full_nums,
            "label": full_labels
        }
        
        with open(cache_path, "wb") as f:
            pickle.dump(data_dict, f)
            
        print(f">> Cache saved to {cache_path} (Total events: {len(full_labels)})")

    def setup(self, stage=None):
        cache_path = os.path.join(self.cache_dir, f"data_cache_len{self.max_particles}.pkl")
        if not os.path.exists(cache_path):
            raise RuntimeError("Cache not found. Run prepare_data first.")
            
        with open(cache_path, "rb") as f:
            data_dict = pickle.load(f)
            
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
        
        self.train_ds = RCNPDataset(data_dict, ind_train)
        self.val_ds = RCNPDataset(data_dict, ind_val)
        self.test_ds = RCNPDataset(data_dict, ind_test)
        
        print(f"Dataset Split: Train={len(self.train_ds)}, Val={len(self.val_ds)}, Test={len(self.test_ds)}")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.conf["batch_size"], 
                          num_workers=self.conf["num_workers"], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.conf["batch_size"], 
                          num_workers=self.conf["num_workers"], shuffle=False)
                          
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.conf["batch_size"], 
                          num_workers=self.conf["num_workers"], shuffle=False)