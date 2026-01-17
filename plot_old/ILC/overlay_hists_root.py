#!/usr/bin/env python
from pathlib import Path
import numpy as np

import ROOT as r
from dataset_utils_jet import load_dataset

# PyROOTのGC対策用：作ったヒストを全部ここに保持しておく
KEEP_HISTS = []

# 描画スタイル
r.gStyle.SetOptStat(1110)  # entries, mean, RMS
r.gStyle.SetStatBorderSize(1)
r.gStyle.SetStatX(0.88)
r.gStyle.SetStatY(0.88)



# Track-level 6個を 2×3 に描画

def plot_track_grid_root(trk_data, class_names, out_path):
    plot_defs = [
        ("E_trk",  "E_trk [GeV]",  "Track energy E_trk"),
        ("Px_trk", "Px_trk [GeV]", "Track momentum Px_trk"),
        ("Py_trk", "Py_trk [GeV]", "Track momentum Py_trk"),
        ("Pz_trk", "Pz_trk [GeV]", "Track momentum Pz_trk"),
        ("P_trk",  "P_trk [GeV]",  "Track momentum |P_trk|"),
        ("Pt_trk", "Pt_trk [GeV]", "Track transverse momentum Pt_trk"),
    ]

    class_order = [0, 1, 2]
    colors = {
        0: r.kBlue,
        1: r.kOrange + 1,
        2: r.kGreen + 2,
    }

    c = r.TCanvas("c_tracks", "Track-level overlay", 1800, 1000)
    c.Divide(3, 2)

    for idx, (var, xlabel, title) in enumerate(plot_defs):
        pad = c.cd(idx + 1)

        # 全クラス合わせて範囲を決める
        all_vals = np.concatenate([np.array(trk_data[q][var]) for q in class_order])
        if all_vals.size == 0:
            continue

        vmin = float(all_vals.min())
        vmax = float(all_vals.max())
        if vmin == vmax:
            vmin -= 1.0
            vmax += 1.0
        margin = 0.05 * (vmax - vmin)
        vmin -= margin
        vmax += margin

        nbins = 100
        hists = {}

        # クラスごとのヒストを作成
        for q in class_order:
            name = f"h_{var}_{class_names[q]}_{idx}"
            h = r.TH1F(name, title, nbins, vmin, vmax)
            h.SetLineColor(colors[q])
            h.SetLineWidth(2)
            h.SetStats(True)

            vals = np.array(trk_data[q][var], dtype=np.float64)
            for v in vals:
                h.Fill(float(v))

            hists[q] = h
            KEEP_HISTS.append(h)  # GC対策で保持

        # 描画
        pad.cd()
        first = True
        for j, q in enumerate(class_order):
            opt = "HIST" if first else "HIST SAME"
            first = False
            hists[q].GetXaxis().SetTitle(xlabel)
            hists[q].GetYaxis().SetTitle("Entries")
            hists[q].Draw(opt)
            pad.Update()

            # Stats box の位置調整
            st = hists[q].GetListOfFunctions().FindObject("stats")
            if st:
                st.SetTextColor(colors[q])
                st.SetLineColor(colors[q])
                y2 = 0.88 - 0.18 * j
                y1 = y2 - 0.16
                st.SetX1NDC(0.60)
                st.SetX2NDC(0.89)
                st.SetY1NDC(y1)
                st.SetY2NDC(y2)
                pad.Modified()

        # 凡例
        leg = r.TLegend(0.15, 0.75, 0.45, 0.90)
        for q in class_order:
            leg.AddEntry(hists[q], class_names[q], "l")
        leg.Draw()
        pad.Modified()

    c.Update()
    c.SaveAs(str(out_path))
    print(f"[Saved] {out_path}")



# Jet-level 6個 + M_jet + trk_jet を 3×3 に描画
def plot_jet_grid_root(jet_data, M_jet_by_cls, ntrk_jet_by_cls, class_names, out_path):
    plot_defs = [
        ("E_jet",   "E_jet [GeV]",   "Jet energy E_jet",               "jet"),
        ("Px_jet",  "Px_jet [GeV]",  "Jet momentum Px_jet",            "jet"),
        ("Py_jet",  "Py_jet [GeV]",  "Jet momentum Py_jet",            "jet"),
        ("Pz_jet",  "Pz_jet [GeV]",  "Jet momentum Pz_jet",            "jet"),
        ("P_jet",   "P_jet [GeV]",   "Jet momentum |P_jet|",           "jet"),
        ("Pt_jet",  "Pt_jet [GeV]",  "Jet transverse momentum Pt_jet", "jet"),
        ("M_jet",   "M_jet [GeV]",   "Jet mass M_jet",                 "mass"),
        ("trk_jet", "trk_jet",       "Tracks per jet (trk_jet)",       "ntrk"),
    ]

    class_order = [0, 1, 2]
    colors = {
        0: r.kBlue,
        1: r.kOrange + 1,
        2: r.kGreen + 2,
    }

    c = r.TCanvas("c_jets", "Jet-level overlay", 1800, 1200)
    c.Divide(3, 3)

    # trk_jet 用の共通ビンを決める（整数）
    all_ntrk = np.concatenate([np.array(ntrk_jet_by_cls[q]) for q in class_order])
    if all_ntrk.size > 0:
        ntrk_min = int(all_ntrk.min())
        ntrk_max = int(all_ntrk.max())
        ntrk_nbins = ntrk_max - ntrk_min + 1
        ntrk_x_min = ntrk_min - 0.5
        ntrk_x_max = ntrk_max + 0.5
    else:
        ntrk_nbins = 1
        ntrk_x_min = -0.5
        ntrk_x_max = 0.5

    for idx, (var, xlabel, title, kind) in enumerate(plot_defs):
        pad = c.cd(idx + 1)

        # データ取得
        if kind == "jet":
            arrays = [np.array(jet_data[q][var]) for q in class_order]
        elif kind == "mass":
            arrays = [np.array(M_jet_by_cls[q]) for q in class_order]
        elif kind == "ntrk":
            arrays = [np.array(ntrk_jet_by_cls[q]) for q in class_order]
        else:
            continue

        all_vals = np.concatenate(arrays) if arrays and arrays[0].size > 0 else np.array([])
        if all_vals.size == 0:
            continue

        # ビン設定
        if kind == "ntrk":
            nbins = ntrk_nbins
            x_min = ntrk_x_min
            x_max = ntrk_x_max
        else:
            vmin = float(all_vals.min())
            vmax = float(all_vals.max())
            if vmin == vmax:
                vmin -= 1.0
                vmax += 1.0
            margin = 0.05 * (vmax - vmin)
            x_min = vmin - margin
            x_max = vmax + margin
            nbins = 100

        hists = {}
        for q, arr in zip(class_order, arrays):
            name = f"h_{var}_{class_names[q]}_{idx}"
            h = r.TH1F(name, title, nbins, x_min, x_max)
            h.SetLineColor(colors[q])
            h.SetLineWidth(2)
            h.SetStats(True)

            for v in arr:
                h.Fill(float(v))

            hists[q] = h
            KEEP_HISTS.append(h)  # GC対策で保持

        # 描画
        pad.cd()
        first = True
        for j, q in enumerate(class_order):
            opt = "HIST" if first else "HIST SAME"
            first = False
            hists[q].GetXaxis().SetTitle(xlabel)
            hists[q].GetYaxis().SetTitle("Entries")
            hists[q].Draw(opt)
            pad.Update()

            # Stats box をずらす
            st = hists[q].GetListOfFunctions().FindObject("stats")
            if st:
                st.SetTextColor(colors[q])
                st.SetLineColor(colors[q])
                y2 = 0.88 - 0.18 * j
                y1 = y2 - 0.16
                st.SetX1NDC(0.60)
                st.SetX2NDC(0.89)
                st.SetY1NDC(y1)
                st.SetY2NDC(y2)
                pad.Modified()

        # 凡例
        leg = r.TLegend(0.15, 0.75, 0.45, 0.90)
        for q in class_order:
            leg.AddEntry(hists[q], class_names[q], "l")
        leg.Draw()
        pad.Modified()

    # 9マス中 8マス使用なので、最後の1マスを空にする
    last_idx = len(plot_defs)
    if last_idx < 9:
        pad = c.cd(last_idx + 1)
        pad.Clear()

    c.Update()
    c.SaveAs(str(out_path))
    print(f"[Saved] {out_path}")



# Main
def main():
    print("Loading jet-level dataset (from dataset_utils_jet)...")
    dat, num, lab = load_dataset()
    n_entries = len(dat)
    print(f"Jets (entries): {n_entries}")

    out_dir = Path("histograms_root/overlay")
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
    ntrk_jet_by_cls = {c: [] for c in class_labels}

    for i in range(n_entries):
        label = int(lab[i])
        n_trk = int(num[i])
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
            mass2 = 0.0
        M_jet_by_cls[label].append(np.sqrt(mass2))

    # Track 2×3, Jet+M_jet+trk_jet 3×3 を ROOT で出力
    plot_track_grid_root(
        trk_data, class_names,
        out_dir / "track_overlay_root_2x3.png"
    )
    plot_jet_grid_root(
        jet_data, M_jet_by_cls, ntrk_jet_by_cls, class_names,
        out_dir / "jet_overlay_root_3x3.png"
    )


if __name__ == "__main__":
    main()
