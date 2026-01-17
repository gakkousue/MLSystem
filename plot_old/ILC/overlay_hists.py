#!/usr/bin/env python
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from dataset_utils import load_dataset

# Track-level 6個を 2×3 にまとめて描画

def plot_track_grid(trk_data, class_names, out_path):
    # (変数名, xラベル, タイトル)
    plot_defs = [
        ("E_trk",  "E_trk [GeV]",  "Track energy E_trk"),
        ("Px_trk", "Px_trk [GeV]", "Track momentum Px_trk"),
        ("Py_trk", "Py_trk [GeV]", "Track momentum Py_trk"),
        ("Pz_trk", "Pz_trk [GeV]", "Track momentum Pz_trk"),
        ("P_trk",  "P_trk [GeV]",  "Track momentum |P_trk|"),
        ("Pt_trk", "Pt_trk [GeV]", "Track transverse momentum Pt_trk"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    colors = ["blue", "orange", "green"]
    cls_order = [0, 1, 2]  # bb, cc, uds
    labels = [class_names[c] for c in cls_order]

    for idx, (var, xlabel, title) in enumerate(plot_defs):
        r = idx // 3
        c = idx % 3
        ax = axes[r, c]

        data_list = [np.array(trk_data[cl][var]) for cl in cls_order]
        # 共通ビン（100）で重ねる
        ax.hist(
            data_list,
            bins=100,
            histtype="step",
            color=colors,
            label=labels,
            linewidth=1.2,
        )
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Entries")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    # 凡例は左上のパネルにまとめて表示
    axes[0, 0].legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved] {out_path}")



# Jet-level 6個＋M_jet を 3×3 にまとめて描画

def plot_jet_grid(jet_data, M_jet_by_cls, class_names, out_path):
    # 最初の6つは jet_data から，最後の1つは M_jet から取る
    plot_defs = [
        ("E_jet",  "E_jet [GeV]",  "Jet energy E_jet",  "jet"),
        ("Px_jet", "Px_jet [GeV]", "Jet momentum Px_jet", "jet"),
        ("Py_jet", "Py_jet [GeV]", "Jet momentum Py_jet", "jet"),
        ("Pz_jet", "Pz_jet [GeV]", "Jet momentum Pz_jet", "jet"),
        ("P_jet",  "P_jet [GeV]",  "Jet momentum |P_jet|", "jet"),
        ("Pt_jet", "Pt_jet [GeV]", "Jet transverse momentum Pt_jet", "jet"),
        ("M_jet",  "M_jet [GeV]",  "Jet mass M_jet", "mass"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    colors = ["blue", "orange", "green"]
    cls_order = [0, 1, 2]
    labels = [class_names[c] for c in cls_order]

    for idx, (var, xlabel, title, kind) in enumerate(plot_defs):
        r = idx // 3
        c = idx % 3
        ax = axes[r, c]

        if kind == "jet":
            data_list = [np.array(jet_data[cl][var]) for cl in cls_order]
        else:  # "mass"
            data_list = [np.array(M_jet_by_cls[cl]) for cl in cls_order]

        ax.hist(
            data_list,
            bins=100,
            histtype="step",
            color=colors,
            label=labels,
            linewidth=1.2,
        )
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Entries")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    # 使っていない残り2パネルは非表示
    for idx in range(len(plot_defs), 9):
        r = idx // 3
        c = idx % 3
        axes[r, c].axis("off")

    axes[0, 0].legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved] {out_path}")


def main():
    # データ読み込み
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

    # 出力フォルダ: histograms/overlay/
    root_dir = Path("histograms")
    out_dir = root_dir / "overlay"
    out_dir.mkdir(parents=True, exist_ok=True)

    # カラム定義
    col_E = 0
    col_Px = 1
    col_Py = 2
    col_Pz = 3

    # クラス情報
    class_labels = [0, 1, 2]  # 0:bb,1:cc,2:uds
    class_names = {0: "bb", 1: "cc", 2: "uds"}

    # track-level
    trk_vars = ["E_trk", "Px_trk", "Py_trk", "Pz_trk", "P_trk", "Pt_trk"]
    trk_data = {c: {v: [] for v in trk_vars} for c in class_labels}

    # jet-level
    jet_vars = ["E_jet", "Px_jet", "Py_jet", "Pz_jet", "P_jet", "Pt_jet"]
    jet_data = {c: {v: [] for v in jet_vars} for c in class_labels}

    # M_jet
    M_jet_by_cls = {c: [] for c in class_labels}

    # イベントループ
    for i in range(n_events):
        label = int(lab[i])
        n_trk = int(num[i])
        parts = dat[i]["part"][:n_trk, :]

        if n_trk == 0:
            continue

        E = parts[:, col_E]
        px = parts[:, col_Px]
        py = parts[:, col_Py]
        pz = parts[:, col_Pz]

        p = np.sqrt(px**2 + py**2 + pz**2)
        pt = np.sqrt(px**2 + py**2)

        # track-level
        trk_data[label]["E_trk"].extend(E.tolist())
        trk_data[label]["Px_trk"].extend(px.tolist())
        trk_data[label]["Py_trk"].extend(py.tolist())
        trk_data[label]["Pz_trk"].extend(pz.tolist())
        trk_data[label]["P_trk"].extend(p.tolist())
        trk_data[label]["Pt_trk"].extend(pt.tolist())

        # jet-level
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

        mass2 = E_jet**2 - P_jet**2
        if mass2 < 0:
            mass2 = 0.0
        M_jet = np.sqrt(mass2)
        M_jet_by_cls[label].append(M_jet)

    #  グリッド図の出力
    track_grid_path = out_dir / "overlay_tracks_2x3.png"
    plot_track_grid(trk_data, class_names, track_grid_path)

    jet_grid_path = out_dir / "overlay_jets_3x3.png"
    plot_jet_grid(jet_data, M_jet_by_cls, class_names, jet_grid_path)

    print("All overlay grid histograms have been generated.")


if __name__ == "__main__":
    main()
