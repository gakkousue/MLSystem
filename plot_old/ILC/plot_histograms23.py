#!/usr/bin/env python
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 画面がない環境でもOK
import matplotlib.pyplot as plt



# データ読み込み（元の学習コードと同じ形式）

from dataset_utils import load_dataset



# 3×2 ヒストを 1 枚の figure として描画するユーティリティ

def plot_3x2_pair(
    data1_by_cls,
    data2_by_cls,
    class_labels,
    class_names,
    xlabel1,
    xlabel2,
    title1,
    title2,
    out_path,
    is_int_bins1=False,
    bins1=50,
    bins2=50,
    logy1=False,
    logy2=True,
):
    """
    data1_by_cls, data2_by_cls: {label: list[...] }
    class_labels: [0,1,2]
    class_names: {0:"bb",1:"cc",2:"uds"}
    """

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))  # 3行×2列（縦長）

    # ---- ビン設定（共通ビンをクラス横断で決める） ----
    # 1列目
    all1 = np.concatenate([np.array(data1_by_cls[c]) for c in class_labels])
    if is_int_bins1:
        vmin, vmax = int(all1.min()), int(all1.max())
        bins_1 = np.arange(vmin - 0.5, vmax + 1.5, 1.0)
    else:
        bins_1 = bins1

    # 2列目
    all2 = np.concatenate([np.array(data2_by_cls[c]) for c in class_labels])
    bins_2 = bins2

    for i, c in enumerate(class_labels):
        cname = class_names[c]

        # 左列：data1
        ax_left = axes[i, 0]
        ax_left.hist(
            data1_by_cls[c],
            bins=bins_1,
            histtype="step",
        )
        if logy1:
            ax_left.set_yscale("log")
        ax_left.grid(True)
        if i == 0:
            ax_left.set_title(title1)
        if i == 2:
            ax_left.set_xlabel(xlabel1)
        ax_left.set_ylabel(f"{cname} (Entries)")

        # 右列：data2
        ax_right = axes[i, 1]
        ax_right.hist(
            data2_by_cls[c],
            bins=bins_2,
            histtype="step",
        )
        if logy2:
            ax_right.set_yscale("log")
        ax_right.grid(True)
        if i == 0:
            ax_right.set_title(title2)
        if i == 2:
            ax_right.set_xlabel(xlabel2)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    
    # 1. データ読み込み
    
    data_directory = Path("../data/ILC.2019.09-low-level-data/")
    fnames = [
        str(data_directory / "bb_data1.txt"),
        str(data_directory / "cc_data1.txt"),
        str(data_directory / "uds_data1.txt"),
    ]

    print("Loading dataset...")
    dat, num, lab = load_dataset(fnames, sanity_check=False)
    n_events = len(dat)
    print(f"Number of events = {n_events}")

    out_dir = Path("histograms")
    out_dir.mkdir(exist_ok=True)

    # part のカラム定義
    col_E = 0   # E_trk
    col_Px = 1  # Px_trk
    col_Py = 2  # Py_trk
    col_Pz = 3  # Pz_trk

    # クラスラベルと名前
    class_labels = [0, 1, 2]
    class_names = {0: "bb", 1: "cc", 2: "uds"}

    
    # 2. クラス別の分布格納用辞書
    
    # track-level
    trk_vars = ["E_trk", "Px_trk", "Py_trk", "Pz_trk", "P_trk", "Pt_trk"]
    trk_data = {c: {v: [] for v in trk_vars} for c in class_labels}

    # jet-level
    jet_vars = ["E_jet", "Px_jet", "Py_jet", "Pz_jet", "P_jet", "Pt_jet"]
    jet_data = {c: {v: [] for v in jet_vars} for c in class_labels}

    # special: #trk_jet と M_jet
    ntrk_jet_by_cls = {c: [] for c in class_labels}
    M_jet_by_cls = {c: [] for c in class_labels}

    
    # 3. イベントループで物理量を計算・格納
    
    for i in range(n_events):
        label = int(lab[i])  # 0,1,2
        n_trk = int(num[i])
        ntrk_jet_by_cls[label].append(n_trk)

        parts = dat[i]["part"][:n_trk, :]  # shape: (n_trk, 7)
        if n_trk == 0:
            continue

        E = parts[:, col_E]
        px = parts[:, col_Px]
        py = parts[:, col_Py]
        pz = parts[:, col_Pz]

        p = np.sqrt(px**2 + py**2 + pz**2)
        pt = np.sqrt(px**2 + py**2)

        # track-level の格納
        trk_data[label]["E_trk"].extend(E.tolist())
        trk_data[label]["Px_trk"].extend(px.tolist())
        trk_data[label]["Py_trk"].extend(py.tolist())
        trk_data[label]["Pz_trk"].extend(pz.tolist())
        trk_data[label]["P_trk"].extend(p.tolist())
        trk_data[label]["Pt_trk"].extend(pt.tolist())

        # jet-level 量
        E_jet = np.sum(E)
        Px_jet = np.sum(px)
        Py_jet = np.sum(py)
        Pz_jet = np.sum(pz)

        P_jet = np.sqrt(Px_jet**2 + Py_jet**2 + Pz_jet**2)
        Pt_jet = np.sqrt(Px_jet**2 + Py_jet**2)

        jet_data[label]["E_jet"].append(E_jet)
        jet_data[label]["Px_jet"].append(Px_jet)
        jet_data[label]["Py_jet"].append(Py_jet)
        jet_data[label]["Pz_jet"].append(Pz_jet)
        jet_data[label]["P_jet"].append(P_jet)
        jet_data[label]["Pt_jet"].append(Pt_jet)

        # M_jet
        mass2 = E_jet**2 - P_jet**2
        if mass2 < 0:
            mass2 = 0.0
        M_jet = np.sqrt(mass2)
        M_jet_by_cls[label].append(M_jet)

    
    # 4. 7 枚の 3×2 ヒスト図を作成
    

    # (1) trk_jet vs M_jet
    plot_3x2_pair(
        data1_by_cls=ntrk_jet_by_cls,
        data2_by_cls=M_jet_by_cls,
        class_labels=class_labels,
        class_names=class_names,
        xlabel1="trk_jet (tracks per jet)",
        xlabel2="M_jet [GeV]",
        title1="trk_jet distribution",
        title2="Jet mass M_jet",
        out_path=out_dir / "hist_ntrkjet_Mjet_by_flavor_3x2.png",
        is_int_bins1=True,
        bins1=50,
        bins2=50,
        logy1=False,
        logy2=True,
    )

    # (2) E_trk vs E_jet
    plot_3x2_pair(
        data1_by_cls={c: trk_data[c]["E_trk"] for c in class_labels},
        data2_by_cls={c: jet_data[c]["E_jet"] for c in class_labels},
        class_labels=class_labels,
        class_names=class_names,
        xlabel1="E_trk [GeV]",
        xlabel2="E_jet [GeV]",
        title1="Track energy E_trk",
        title2="Jet energy E_jet",
        out_path=out_dir / "hist_Etrk_Ejet_by_flavor_3x2.png",
        is_int_bins1=False,
        bins1=100,
        bins2=100,
        logy1=True,
        logy2=True,
    )

    # (3) Px_trk vs Px_jet
    plot_3x2_pair(
        data1_by_cls={c: trk_data[c]["Px_trk"] for c in class_labels},
        data2_by_cls={c: jet_data[c]["Px_jet"] for c in class_labels},
        class_labels=class_labels,
        class_names=class_names,
        xlabel1="Px_trk [GeV]",
        xlabel2="Px_jet [GeV]",
        title1="Track momentum Px_trk",
        title2="Jet momentum Px_jet",
        out_path=out_dir / "hist_Pxtrk_Pxjet_by_flavor_3x2.png",
        is_int_bins1=False,
        bins1=100,
        bins2=100,
        logy1=True,
        logy2=True,
    )

    # (4) Py_trk vs Py_jet
    plot_3x2_pair(
        data1_by_cls={c: trk_data[c]["Py_trk"] for c in class_labels},
        data2_by_cls={c: jet_data[c]["Py_jet"] for c in class_labels},
        class_labels=class_labels,
        class_names=class_names,
        xlabel1="Py_trk [GeV]",
        xlabel2="Py_jet [GeV]",
        title1="Track momentum Py_trk",
        title2="Jet momentum Py_jet",
        out_path=out_dir / "hist_Pytrk_Pyjet_by_flavor_3x2.png",
        is_int_bins1=False,
        bins1=100,
        bins2=100,
        logy1=True,
        logy2=True,
    )

    # (5) Pz_trk vs Pz_jet
    plot_3x2_pair(
        data1_by_cls={c: trk_data[c]["Pz_trk"] for c in class_labels},
        data2_by_cls={c: jet_data[c]["Pz_jet"] for c in class_labels},
        class_labels=class_labels,
        class_names=class_names,
        xlabel1="Pz_trk [GeV]",
        xlabel2="Pz_jet [GeV]",
        title1="Track momentum Pz_trk",
        title2="Jet momentum Pz_jet",
        out_path=out_dir / "hist_Pztrk_Pzjet_by_flavor_3x2.png",
        is_int_bins1=False,
        bins1=100,
        bins2=100,
        logy1=True,
        logy2=True,
    )

    # (6) P_trk vs P_jet
    plot_3x2_pair(
        data1_by_cls={c: trk_data[c]["P_trk"] for c in class_labels},
        data2_by_cls={c: jet_data[c]["P_jet"] for c in class_labels},
        class_labels=class_labels,
        class_names=class_names,
        xlabel1="P_trk [GeV]",
        xlabel2="P_jet [GeV]",
        title1="Track momentum P_trk",
        title2="Jet momentum P_jet",
        out_path=out_dir / "hist_Ptrk_Pjet_by_flavor_3x2.png",
        is_int_bins1=False,
        bins1=100,
        bins2=100,
        logy1=True,
        logy2=True,
    )

    # (7) Pt_trk vs Pt_jet
    plot_3x2_pair(
        data1_by_cls={c: trk_data[c]["Pt_trk"] for c in class_labels},
        data2_by_cls={c: jet_data[c]["Pt_jet"] for c in class_labels},
        class_labels=class_labels,
        class_names=class_names,
        xlabel1="Pt_trk [GeV]",
        xlabel2="Pt_jet [GeV]",
        title1="Track transverse momentum Pt_trk",
        title2="Jet transverse momentum Pt_jet",
        out_path=out_dir / "hist_Pttrk_Ptjet_by_flavor_3x2.png",
        is_int_bins1=False,
        bins1=100,
        bins2=100,
        logy1=True,
        logy2=True,
    )

    print("All 3x2 histograms generated.")


if __name__ == "__main__":
    main()
