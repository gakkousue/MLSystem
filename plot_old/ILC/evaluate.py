#!/usr/bin/env python
from pathlib import Path

import numpy as np
import random
import pickle
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



from dataset_utils import load_dataset

#  RCNPDataset クラス

class RCNPDataset(Dataset):
    def __init__(self, dat, num, lab, split='test'):
        if split == 'test':
            self.dat = dat
            self.num = np.array(num, dtype=np.float32)
            self.lab = lab
        else:
            raise ValueError("This file is for evaluation only. split must be 'test'.")

    def __len__(self):
        return len(self.dat)
    
    def __getitem__(self, i):
        axis = self.dat[i]['axis']
        part = self.dat[i]['part']
        num  = self.num[i]
        lab  = self.lab[i]
        return axis, part, num, lab



#  Baseline Model

class BaselineModel(nn.Module):
    def __init__(self, n_units=100, n_out=3):
        super(BaselineModel, self).__init__()
        self.lp1 = nn.Linear(7, n_units)
        self.lp2 = nn.Linear(n_units, n_units)
        self.lc1 = nn.Linear(n_units + 3, n_units)
        self.lc2 = nn.Linear(n_units, n_units)
        self.lc3 = nn.Linear(n_units, n_units)
        self.lc4 = nn.Linear(n_units, n_units)
        self.lc5 = nn.Linear(n_units, n_out)

    def forward(self, a, p, n):
        n_bat, n_par, n_dim = p.shape 
        nonl = torch.relu
        
        h = p.view(-1, n_dim)
        h = nonl(self.lp1(h))
        h = nonl(self.lp2(h))
        h = h.view(n_bat, n_par, -1)
        
        h = torch.sum(h, dim=1) / n.view(-1, 1)
        h = torch.cat((h, a), dim=1)

        h = nonl(self.lc1(h))
        h = nonl(self.lc2(h))
        h = nonl(self.lc3(h))
        h = nonl(self.lc4(h))
        return self.lc5(h)



#  ROC (0 対策入り)

def calculate_roc_points(t, y_prob, signal_label, background_label):
    signal_probs = y_prob[t == signal_label, signal_label]
    background_probs = y_prob[t == background_label, signal_label]
    
    n_signal = len(signal_probs)
    n_background = len(background_probs)

    thresholds = np.linspace(0, 1, 1001)
    
    efficiencies = []
    misids = []
    
    for th in thresholds:
        eff = np.sum(signal_probs >= th) / n_signal if n_signal > 0 else 0.
        misid = np.sum(background_probs >= th) / n_background if n_background > 0 else 0.

        # ログ軸用の擬似値
        if misid == 0 and n_background > 0:
            misid = 0.5 / n_background

        efficiencies.append(eff)
        misids.append(misid)
        
    return np.array(efficiencies), np.array(misids)



#   EVALUATE ONLY

def evaluate_only(weight_path, n_units=100):
    device = torch.device("cpu")

    #  データ読み込み 
    data_directory = Path('../data/ILC.2019.09-low-level-data/')
    with open(data_directory / 'rcnpdataset_test.pickle', 'rb') as fin:
        ds_test = pickle.load(fin)

    loader = DataLoader(ds_test, batch_size=2048, shuffle=False)

    #  モデル読み込み 
    model = BaselineModel(n_units=n_units, n_out=3)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    #  推論 
    y_list = []
    t_list = []

    with torch.no_grad():
        for a, p, n, t in loader:
            a, p, n = a.float().to(device), p.float().to(device), n.float().to(device)
            outputs = model(a, p, n)
            prob = F.softmax(outputs, dim=1)

            y_list.append(prob.cpu().numpy())
            t_list.append(t.numpy())

    y = np.concatenate(y_list, axis=0)
    t = np.concatenate(t_list, axis=0)

    # 保存先
    output_dir = Path(weight_path).parent

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(t, np.argmax(y, axis=1))

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axis("off")

    table = ax.table(
        cellText=cm,
        rowLabels=["bb event", "cc event", "uds event"],
        colLabels=["b", "c", "uds"],
        cellLoc="center",
        loc="center",
    )
    table.scale(1.2, 1.6)
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()


    # Likelihood Histograms
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    class_names = ["bb", "cc", "uds"]
    colors = ["red", "blue", "green"]

    for true_cls in range(3):
        mask = (t == true_cls)
        probs_true = y[mask]

        for pred_cls in range(3):
            ax = axes[true_cls, pred_cls]
            ax.hist(probs_true[:, pred_cls], bins=50, color=colors[true_cls], alpha=0.7, density=True)

            ax.set_xlim(0, 1)
            if pred_cls == 0:
                ax.set_ylabel(f"{class_names[true_cls]} events")
            if true_cls == 2:
                ax.set_xlabel(f"{class_names[pred_cls]} likeness")

            ax.set_title(f"{class_names[true_cls]}-{class_names[pred_cls]}")

    plt.tight_layout()
    plt.savefig(output_dir / "likeness_histograms.png")
    plt.close()


    # ROC
    class_names = ["bb", "cc", "uds"]

    for signal_idx in range(3):
        plt.figure(figsize=(7, 6))

        for background_idx in range(3):
            if signal_idx == background_idx:
                continue

            effs, misids = calculate_roc_points(t, y, signal_idx, background_idx)

            plt.plot(effs, misids, label=f"BG: {class_names[background_idx]}")

        plt.yscale("log")
        plt.grid(True, which="both", linestyle='--')
        plt.xlabel(f"{class_names[signal_idx]} tagging efficiency")
        plt.ylabel("Mis-identification Fraction")
        plt.title(f"ROC Curve for {class_names[signal_idx]}-tagging")
        plt.legend()
        plt.savefig(output_dir / f"roc_{class_names[signal_idx]}_tagging.png")
        plt.close()


#   MAIN
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model only.")
    parser.add_argument("weight", help="Path to model_epoch-XXX.pth")
    parser.add_argument("--n-units", type=int, default=100)
    args = parser.parse_args()

    evaluate_only(args.weight, n_units=args.n_units)
