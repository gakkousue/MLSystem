#!/usr/bin/env python
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 画面がない環境でもOK
import matplotlib.pyplot as plt



from dataset_utils import load_dataset


# ヒストグラム描画用のユーティリティ
def plot_hist(data, bins, xlabel, title, out_path, logy=False):
    plt.figure()
    plt.hist(data, bins=bins, histtype="step")
    plt.xlabel(xlabel)
    plt.ylabel("Entries")
    plt.title(title)
    if logy:
        plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    # 学習コードと同じ data ディレクトリを想定
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

    # 出力ディレクトリ
    out_dir = Path("histograms")
    out_dir.mkdir(exist_ok=True)

    # ここから物理量の計算
    # part のカラム定義（必要ならここを修正）
    col_E = 0   # E_trk
    col_Px = 1  # Px_trk
    col_Py = 2  # Py_trk
    col_Pz = 3  # Pz_trk

    #1) trk_jet （イベントごとのトラック数）
    ntrk_jet_list = []

    #2) Track-level quantities
    E_trk_list = []
    Px_trk_list = []
    Py_trk_list = []
    Pz_trk_list = []
    P_trk_list = []
    Pt_trk_list = []

    #3) Jet-level quantities
    E_jet_list = []
    Px_jet_list = []
    Py_jet_list = []
    Pz_jet_list = []
    P_jet_list = []
    Pt_jet_list = []
    M_jet_list = []

    for i in range(n_events):
        n_trk = int(num[i])
        ntrk_jet_list.append(n_trk)

        parts = dat[i]["part"][:n_trk, :]  # shape: (n_trk, 7)
        if n_trk == 0:
            # 念のため（基本的には 0 にはならないはず）
            continue

        E = parts[:, col_E]
        px = parts[:, col_Px]
        py = parts[:, col_Py]
        pz = parts[:, col_Pz]

        p = np.sqrt(px**2 + py**2 + pz**2)
        pt = np.sqrt(px**2 + py**2)

        # track 単位で保存
        E_trk_list.extend(E.tolist())
        Px_trk_list.extend(px.tolist())
        Py_trk_list.extend(py.tolist())
        Pz_trk_list.extend(pz.tolist())
        P_trk_list.extend(p.tolist())
        Pt_trk_list.extend(pt.tolist())

        # jet 単位の量
        E_jet = np.sum(E)
        Px_jet = np.sum(px)
        Py_jet = np.sum(py)
        Pz_jet = np.sum(pz)

        Pt_jet = np.sqrt(Px_jet**2 + Py_jet**2)
        P_jet = np.sqrt(Px_jet**2 + Py_jet**2 + Pz_jet**2)
        # 数値誤差で負になるのを防ぐ
        mass2 = E_jet**2 - P_jet**2
        if mass2 < 0:
            mass2 = 0.0
        M_jet = np.sqrt(mass2)

        E_jet_list.append(E_jet)
        Px_jet_list.append(Px_jet)
        Py_jet_list.append(Py_jet)
        Pz_jet_list.append(Pz_jet)
        Pt_jet_list.append(Pt_jet)
        P_jet_list.append(P_jet)
        M_jet_list.append(M_jet)

    # ヒストグラム描画

    print("Plotting histograms...")

    #trk_jet
    plot_hist(
        data=ntrk_jet_list,
        bins=50,
        xlabel="trk_jet (tracks per jet)",
        title="trk_jet distribution",
        out_path=out_dir / "hist_ntrk_jet.png",
        logy=False,
    )

    #Track-level
    plot_hist(
        E_trk_list,
        bins=100,
        xlabel="E_trk [GeV]",
        title="Track energy E_trk",
        out_path=out_dir / "hist_E_trk.png",
        logy=True,
    )
    plot_hist(
        Px_trk_list,
        bins=100,
        xlabel="Px_trk [GeV]",
        title="Track momentum Px_trk",
        out_path=out_dir / "hist_Px_trk.png",
        logy=True,
    )
    plot_hist(
        Py_trk_list,
        bins=100,
        xlabel="Py_trk [GeV]",
        title="Track momentum Py_trk",
        out_path=out_dir / "hist_Py_trk.png",
        logy=True,
    )
    plot_hist(
        Pz_trk_list,
        bins=100,
        xlabel="Pz_trk [GeV]",
        title="Track momentum Pz_trk",
        out_path=out_dir / "hist_Pz_trk.png",
        logy=True,
    )
    plot_hist(
        P_trk_list,
        bins=100,
        xlabel="P_trk [GeV]",
        title="Track momentum P_trk",
        out_path=out_dir / "hist_P_trk.png",
        logy=True,
    )
    plot_hist(
        Pt_trk_list,
        bins=100,
        xlabel="Pt_trk [GeV]",
        title="Track transverse momentum Pt_trk",
        out_path=out_dir / "hist_Pt_trk.png",
        logy=True,
    )

    #Jet-level
    plot_hist(
        E_jet_list,
        bins=100,
        xlabel="E_jet [GeV]",
        title="Jet energy E_jet",
        out_path=out_dir / "hist_E_jet.png",
        logy=True,
    )
    plot_hist(
        Px_jet_list,
        bins=100,
        xlabel="Px_jet [GeV]",
        title="Jet momentum Px_jet",
        out_path=out_dir / "hist_Px_jet.png",
        logy=True,
    )
    plot_hist(
        Py_jet_list,
        bins=100,
        xlabel="Py_jet [GeV]",
        title="Jet momentum Py_jet",
        out_path=out_dir / "hist_Py_jet.png",
        logy=True,
    )
    plot_hist(
        Pz_jet_list,
        bins=100,
        xlabel="Pz_jet [GeV]",
        title="Jet momentum Pz_jet",
        out_path=out_dir / "hist_Pz_jet.png",
        logy=True,
    )
    plot_hist(
        P_jet_list,
        bins=100,
        xlabel="P_jet [GeV]",
        title="Jet momentum P_jet",
        out_path=out_dir / "hist_P_jet.png",
        logy=True,
    )
    plot_hist(
        Pt_jet_list,
        bins=100,
        xlabel="Pt_jet [GeV]",
        title="Jet transverse momentum Pt_jet",
        out_path=out_dir / "hist_Pt_jet.png",
        logy=True,
    )
    plot_hist(
        M_jet_list,
        bins=100,
        xlabel="M_jet [GeV]",
        title="Jet mass M_jet",
        out_path=out_dir / "hist_M_jet.png",
        logy=True,
    )

    print(f"Done. PNG files are saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
