#!/usr/bin/env python
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_utils_jet import load_dataset



# Track-level 6個を 2×3 にまとめて描画

def plot_track_grid(trk_data, class_names, out_path):
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
    class_order = [0, 1, 2]

    for idx, (var, xlabel, title) in enumerate(plot_defs):
        r = idx // 3
        c = idx % 3
        ax = axes[r, c]

        data = [np.array(trk_data[q][var]) for q in class_order]

        ax.hist(
            data,
            bins=100,
            histtype="step",
            color=colors,
            linewidth=1.2,
            label=[class_names[q] for q in class_order],
        )
        #ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Entries")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    axes[0, 0].legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved] {out_path}")



# Jet-level 6個 + M_jet + trk_jet を 3×3 にまとめて描画
# （M_jet の隣に trk_jet を配置）

def plot_jet_grid(jet_data, M_jet_by_cls, ntrk_jet_by_cls, class_names, out_path):
    # kind:
    #  - "jet": jet_data から取る量
    #  - "mass": M_jet_by_cls
    #  - "ntrk": ntrk_jet_by_cls
    plot_defs = [
        ("E_jet",    "E_jet [GeV]",    "Jet energy E_jet",              "jet"),
        ("Px_jet",   "Px_jet [GeV]",   "Jet momentum Px_jet",           "jet"),
        ("Py_jet",   "Py_jet [GeV]",   "Jet momentum Py_jet",           "jet"),
        ("Pz_jet",   "Pz_jet [GeV]",   "Jet momentum Pz_jet",           "jet"),
        ("P_jet",    "P_jet [GeV]",    "Jet momentum |P_jet|",          "jet"),
        ("Pt_jet",   "Pt_jet [GeV]",   "Jet transverse momentum Pt_jet","jet"),
        ("M_jet",    "M_jet [GeV]",    "Jet mass M_jet",                "mass"),
        ("trk_jet",  "trk_jet",       "Tracks per jet (trk_jet)",     "ntrk"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    colors = ["blue", "orange", "green"]
    class_order = [0, 1, 2]

    # まず trk_jet のビンは整数になるように共通ビンを作っておく
    all_ntrk = np.concatenate(
        [np.array(ntrk_jet_by_cls[c]) for c in class_order]
    )
    ntrk_min = int(all_ntrk.min())
    ntrk_max = int(all_ntrk.max())
    ntrk_bins = np.arange(ntrk_min - 0.5, ntrk_max + 1.5, 1.0)

    for idx, (var, xlabel, title, kind) in enumerate(plot_defs):
        r = idx // 3
        c = idx % 3
        ax = axes[r, c]

        if kind == "jet":
            data = [np.array(jet_data[q][var]) for q in class_order]
            bins = 100
        elif kind == "mass":
            data = [np.array(M_jet_by_cls[q]) for q in class_order]
            bins = 100
        elif kind == "ntrk":
            data = [np.array(ntrk_jet_by_cls[q]) for q in class_order]
            bins = ntrk_bins
        else:
            continue

        ax.hist(
            data,
            bins=bins,
            histtype="step",
            color=colors,
            linewidth=1.2,
            label=[class_names[q] for q in class_order],
        )
        # ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Entries")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # 残り1パネルを空にする（9マス中8マス使用）
    last_idx = len(plot_defs)
    if last_idx < 9:
        r = last_idx // 3
        c = last_idx % 3
        axes[r, c].axis("off")

    axes[0, 0].legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved] {out_path}")



# Main
def main():
    print("Loading dataset...")
    dat, num, lab = load_dataset()
    n_events = len(dat)
    print(f"Events: {n_events}")

    out_dir = Path("histograms/overlay")
    out_dir.mkdir(parents=True, exist_ok=True)

    col_E = 0
    col_Px = 1
    col_Py = 2
    col_Pz = 3

    class_names = {0: "bb", 1: "cc", 2: "uds"}
    class_labels = [0, 1, 2]

    trk_vars = ["E_trk", "Px_trk", "Py_trk", "Pz_trk", "P_trk", "Pt_trk"]
    jet_vars = ["E_jet", "Px_jet", "Py_jet", "Pz_jet", "P_jet", "Pt_jet"]

    trk_data = {c: {v: [] for v in trk_vars} for c in class_labels}
    jet_data = {c: {v: [] for v in jet_vars} for c in class_labels}
    M_jet_by_cls = {c: [] for c in class_labels}
    ntrk_jet_by_cls = {c: [] for c in class_labels}  # ★ 追加：trk_jet 用

    for i in range(n_events):
        label = int(lab[i])
        n_trk = int(num[i])

        # trk_jet は n_trk が 0 のときもそのまま記録しておく
        ntrk_jet_by_cls[label].append(n_trk)

        parts = dat[i]["part"][:n_trk, :]
        if n_trk == 0:
            continue

        E = parts[:, col_E]
        px = parts[:, col_Px]
        py = parts[:, col_Py]
        pz = parts[:, col_Pz]

        p = np.sqrt(px**2 + py**2 + pz**2)
        pt = np.sqrt(px**2 + py**2)

        # Track-level
        trk_data[label]["E_trk"].extend(E)
        trk_data[label]["Px_trk"].extend(px)
        trk_data[label]["Py_trk"].extend(py)
        trk_data[label]["Pz_trk"].extend(pz)
        trk_data[label]["P_trk"].extend(p)
        trk_data[label]["Pt_trk"].extend(pt)

        # Jet-level
        E_jet = E.sum()
        Px_jet = px.sum()
        Py_jet = py.sum()
        Pz_jet = pz.sum()

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
            mass2 = 0
        M_jet_by_cls[label].append(np.sqrt(mass2))

    # 出力（Track 2×3, Jet+M_jet+trk_jet 3×3）
    plot_track_grid(trk_data, class_names, out_dir / "track_overlay_liner_2x3.png")
    plot_jet_grid(jet_data, M_jet_by_cls, ntrk_jet_by_cls,
                  class_names, out_dir / "jet_overlay_liner_3x3.png")


if __name__ == "__main__":
    main()
