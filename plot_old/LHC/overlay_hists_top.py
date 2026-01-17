#!/usr/bin/env python
from pathlib import Path
import numpy as np
import ROOT as r

from preprocess2 import dump_jets  # alljets をロード

# 1200000 (exactly 1211000) top 605477, qcd 605523
infile_train = "train.h5"
# 400000 (exactly 404000) top 202086, qcd 201914
# infile_test = "../test.h5"
# 400000 (exactly 403000) top 201497, qcd 201503
# infile_val = "../val.h5"

# max_trks = 200

inputs = [[infile_train, 10000, "data_top_train_10k.h5", "data_qcd_train_10k.h5"],]

# PyROOT の GC 対策
KEEP_HISTS = []

# 描画スタイル
r.gStyle.SetOptStat(1110)  # entries, mean, RMS
r.gStyle.SetStatBorderSize(1)
r.gStyle.SetStatX(0.88)
r.gStyle.SetStatY(0.88)



#  alljets -> track / jet 物理量の配列に変換


def build_data_from_alljets(alljets):
    """
    dump_jets() が作った alljets(dict) から、
    track/jet 物理量の配列を復元する。

    dump_jets 側の仕様（ユーザー提示）：
      - jet[:800] が track 部（200 tracks * 4要素）
      - 4要素は (E, px, py, pz)
      - n_tracks は jet[:800:4] の E != 0 の本数
      - track_info は [E,px,py,pz] を n_tracks 回 extend したフラット配列
    """

    # Track-level で扱う変数
    trk_vars = ["E_trk", "Px_trk", "Py_trk", "Pz_trk", "P_trk", "Pt_trk"]

    # Jet-level で扱う変数
    jet_vars = ["E_jet", "Px_jet", "Py_jet", "Pz_jet", "P_jet", "Pt_jet"]

    classes = ["top", "qcd"]

    trk_data = {cls: {v: [] for v in trk_vars} for cls in classes}
    jet_data = {cls: {v: [] for v in jet_vars} for cls in classes}
    M_jet_by_cls = {cls: [] for cls in classes}
    ntrk_jet_by_cls = {cls: [] for cls in classes}

    count = 0

    for cls in classes:
        jets_cls = alljets.get(cls, [])
        for jet_info in jets_cls:
            count += 1
            if len(jet_info) < 7:
                continue

            # ヘッダ
            jet_id, E_jet, px_jet, py_jet, pz_jet, mass_jet, n_tracks = jet_info[:7]
            jet_id = int(jet_id)
            n_tracks = int(n_tracks)

            E_jet = float(E_jet)
            px_jet = float(px_jet)
            py_jet = float(py_jet)
            pz_jet = float(pz_jet)
            mass_jet = float(mass_jet)

            # Jet-level 量
            P_jet = np.sqrt(px_jet**2 + py_jet**2 + pz_jet**2)
            Pt_jet = np.sqrt(px_jet**2 + py_jet**2)

            jet_data[cls]["E_jet"].append(E_jet)
            jet_data[cls]["Px_jet"].append(px_jet)
            jet_data[cls]["Py_jet"].append(py_jet)
            jet_data[cls]["Pz_jet"].append(pz_jet)
            jet_data[cls]["P_jet"].append(P_jet)
            jet_data[cls]["Pt_jet"].append(Pt_jet)

            M_jet_by_cls[cls].append(mass_jet)
            ntrk_jet_by_cls[cls].append(n_tracks)

            # Track-level：dump_jets 仕様に合わせて 4要素固定で読む
            track_flat = jet_info[7:]
            if n_tracks <= 0:
                continue

            # # dump_jets 側では track_info は必ず 4*n_tracks のはず
            # # ただし安全のため short の場合は切り詰める
            # need = 4 * n_tracks
            # if len(track_flat) < need:
            #     # 壊れたイベント：読める範囲に縮める
            #     n_tracks_eff = len(track_flat) // 4
            # else:
            n_tracks_eff = n_tracks

            target_jet_id = 8

            if  jet_id == target_jet_id:
                print(f"[BUILD] cls={cls} jet_id={jet_id} "
                      f"E_jet={E_jet:.6g} px={px_jet:.6g} py={py_jet:.6g} pz={pz_jet:.6g} "
                      f"mass={mass_jet:.6g} n_tracks={n_tracks} (eff={n_tracks_eff})")
                for i in range(min(n_tracks_eff, 20)):
                    print(float(track_flat[4*i + 0]))


            for i_trk in range(n_tracks_eff):
                base = 4 * i_trk
                E_trk  = float(track_flat[base + 0])
                px_trk = float(track_flat[base + 1])
                py_trk = float(track_flat[base + 2])
                pz_trk = float(track_flat[base + 3])

                P_trk  = np.sqrt(px_trk**2 + py_trk**2 + pz_trk**2)
                Pt_trk = np.sqrt(px_trk**2 + py_trk**2)

                trk_data[cls]["E_trk"].append(E_trk)
                trk_data[cls]["Px_trk"].append(px_trk)
                trk_data[cls]["Py_trk"].append(py_trk)
                trk_data[cls]["Pz_trk"].append(pz_trk)
                trk_data[cls]["P_trk"].append(P_trk)
                trk_data[cls]["Pt_trk"].append(Pt_trk)

    return trk_data, jet_data, M_jet_by_cls, ntrk_jet_by_cls



#  Track-level 2×3 overlay

# -------------------------
#  共通：軸範囲をパーセンタイルで決める
# -------------------------

def get_axis_range_percentile(arrays, p_lo=2.0, p_hi=98):
    """
    arrays: [np.ndarray, np.ndarray, ...]  各クラスの値
    p_lo, p_hi: パーセンタイル (0–100)

    外れ値に引きずられないように、全データの p_lo〜p_hi パーセンタイルを
    もとに x_min, x_max を決める。
    """
    all_vals = np.concatenate([a for a in arrays if a.size > 0]) \
        if arrays and arrays[0].size > 0 else np.array([])

    if all_vals.size == 0:
        return None, None

    v_lo = float(np.percentile(all_vals, p_lo))
    v_hi = float(np.percentile(all_vals, p_hi))

    if v_lo == v_hi:
        v_lo -= 1.0
        v_hi += 1.0

    # ちょっとマージンを足す
    margin = 0.05 * (v_hi - v_lo)
    x_min = v_lo - margin
    x_max = v_hi + margin
    return x_min, x_max


# -----------------------------------------
#  Track-level 2×3 overlay (Lab frame 用)
# -----------------------------------------

def plot_track_grid_top_vs_qcd_root(trk_data, out_path):
    """
    trk_data[cls][var] から、top / qcd の Track-level ヒストグラムを
    2×3 キャンバスで overlay 描画する（ラボ系）。
    """

    plot_defs = [
        ("E_trk",  "E_trk [GeV]",  "Track energy E_trk"),
        ("Px_trk", "Px_trk [GeV]", "Track momentum Px_trk"),
        ("Py_trk", "Py_trk [GeV]", "Track momentum Py_trk"),
        ("Pz_trk", "Pz_trk [GeV]", "Track momentum Pz_trk"),
        ("P_trk",  "P_trk [GeV]",  "Track momentum |P_trk|"),
        ("Pt_trk", "Pt_trk [GeV]", "Track transverse momentum Pt_trk"),
    ]

    classes = ["top", "qcd"]
    class_names = {"top": "top", "qcd": "qcd"}
    colors = {
        "top": r.kRed + 1,
        "qcd": r.kBlue + 1,
    }

    c = r.TCanvas("c_tracks_topqcd", "Top vs QCD tracks (lab)", 1800, 1000)
    c.Divide(3, 2)

    for idx, (var, xlabel, title) in enumerate(plot_defs):
        pad = c.cd(idx + 1)

        arrays = [np.array(trk_data[cls][var], dtype=np.float64) for cls in classes]
        x_min, x_max = get_axis_range_percentile(arrays, p_lo=0.5, p_hi=99.5)
        if x_min is None:
            pad.Clear()
            continue

        nbins = 100
        hists = {}

        for cls, arr in zip(classes, arrays):
            name = f"h_{var}_{class_names[cls]}_{idx}"
            h = r.TH1F(name, title, nbins, x_min, x_max)
            h.SetLineColor(colors[cls])
            h.SetLineWidth(2)
            h.SetStats(True)

            for v in arr:
                h.Fill(float(v))

            hists[cls] = h
            KEEP_HISTS.append(h)

        # 描画
        pad.cd()
        first = True
        for j, cls in enumerate(classes):
            opt = "HIST" if first else "HIST SAME"
            first = False
            hists[cls].GetXaxis().SetTitle(xlabel)
            hists[cls].GetYaxis().SetTitle("Entries")
            hists[cls].Draw(opt)
            pad.Update()

            # Stats box の位置調整
            st = hists[cls].GetListOfFunctions().FindObject("stats")
            if st:
                st.SetTextColor(colors[cls])
                st.SetLineColor(colors[cls])
                y2 = 0.88 - 0.20 * j
                y1 = y2 - 0.18
                st.SetX1NDC(0.60)
                st.SetX2NDC(0.89)
                st.SetY1NDC(y1)
                st.SetY2NDC(y2)
                pad.Modified()

        # 凡例
        leg = r.TLegend(0.15, 0.75, 0.45, 0.90)
        for cls in classes:
            leg.AddEntry(hists[cls], class_names[cls], "l")
        leg.Draw()
        pad.Modified()

    c.Update()
    c.SaveAs(str(out_path))
    print(f"[Saved] {out_path}")


# ------------------------------------------------
#  Track-level 2×3 overlay (Jet rest frame 用)
# ------------------------------------------------

def plot_track_grid_top_vs_qcd_root_rest(trk_data_rest, out_path):
    """
    Jet 静止系に Boost 済みの trk_data_rest[cls][var] から、
    top / qcd の Track-level ヒストグラムを 2×3 キャンバスで overlay 描画。
    """

    plot_defs = [
        ("E_trk",  "E_trk [GeV]",  "Track energy E_trk (jet rest)"),
        ("Px_trk", "Px_trk [GeV]", "Track momentum Px_trk (jet rest)"),
        ("Py_trk", "Py_trk [GeV]", "Track momentum Py_trk (jet rest)"),
        ("Pz_trk", "Pz_trk [GeV]", "Track momentum Pz_trk (jet rest)"),
        ("P_trk",  "P_trk [GeV]",  "Track momentum |P_trk| (jet rest)"),
        ("Pt_trk", "Pt_trk [GeV]", "Track transverse momentum Pt_trk (jet rest)"),
    ]

    classes = ["top", "qcd"]
    class_names = {"top": "top", "qcd": "qcd"}
    colors = {
        "top": r.kRed + 1,
        "qcd": r.kBlue + 1,
    }

    c = r.TCanvas("c_tracks_topqcd_rest", "Top vs QCD tracks (jet rest)", 1800, 1000)
    c.Divide(3, 2)

    for idx, (var, xlabel, title) in enumerate(plot_defs):
        pad = c.cd(idx + 1)

        arrays = [np.array(trk_data_rest[cls][var], dtype=np.float64) for cls in classes]
        x_min, x_max = get_axis_range_percentile(arrays, p_lo=0.5, p_hi=99.5)
        if x_min is None:
            pad.Clear()
            continue

        nbins = 100
        hists = {}

        h2 = r.TH1F("E_trk2",  "E_trk [GeV]",  "Track energy E_trk"),

        for cls, arr in zip(classes, arrays):
            name = f"h_{var}_{class_names[cls]}_{idx}_rest"
            h = r.TH1F(name, title, nbins, x_min, x_max)
            h.SetLineColor(colors[cls])
            h.SetLineWidth(2)
            h.SetStats(True)

            for v in arr:
                h.Fill(float(v))

            hists[cls] = h
            KEEP_HISTS.append(h)

        # 描画
        pad.cd()
        first = True
        for j, cls in enumerate(classes):
            opt = "HIST" if first else "HIST SAME"
            first = False
            hists[cls].GetXaxis().SetTitle(xlabel)
            hists[cls].GetYaxis().SetTitle("Entries")
            hists[cls].Draw(opt)
            pad.Update()

            # Stats box の位置調整
            st = hists[cls].GetListOfFunctions().FindObject("stats")
            if st:
                st.SetTextColor(colors[cls])
                st.SetLineColor(colors[cls])
                y2 = 0.88 - 0.20 * j
                y1 = y2 - 0.18
                st.SetX1NDC(0.60)
                st.SetX2NDC(0.89)
                st.SetY1NDC(y1)
                st.SetY2NDC(y2)
                pad.Modified()

        leg = r.TLegend(0.15, 0.75, 0.45, 0.90)
        for cls in classes:
            leg.AddEntry(hists[cls], class_names[cls], "l")
        leg.Draw()
        pad.Modified()

    c.Update()
    c.SaveAs(str(out_path))
    print(f"[Saved] {out_path}")



#  Jet-level 3×3 overlay


def plot_jet_grid_top_vs_qcd_root(jet_data, M_jet_by_cls, ntrk_jet_by_cls, out_path):
    """
    jet_data / M_jet_by_cls / ntrk_jet_by_cls から、
    top / qcd の Jet-level ヒストグラムを 3×3 キャンバスで overlay 描画する。
    """

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

    classes = ["top", "qcd"]
    class_names = {"top": "top", "qcd": "qcd"}
    colors = {
        "top": r.kRed + 1,
        "qcd": r.kBlue + 1,
    }

    c = r.TCanvas("c_jets_topqcd", "Top vs QCD jets", 1800, 1200)
    c.Divide(3, 3)

    # trk_jet 用の共通ビン（整数）
    all_ntrk = []
    for cls in classes:
        all_ntrk.extend(ntrk_jet_by_cls[cls])
    all_ntrk = np.array(all_ntrk, dtype=np.int32)

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

        if kind == "jet":
            arrays = [np.array(jet_data[cls][var], dtype=np.float64) for cls in classes]
        elif kind == "mass":
            arrays = [np.array(M_jet_by_cls[cls], dtype=np.float64) for cls in classes]
        elif kind == "ntrk":
            arrays = [np.array(ntrk_jet_by_cls[cls], dtype=np.float64) for cls in classes]
        else:
            pad.Clear()
            continue
# 400000 (exactly 404000) top 202086, qcd 201914
# infile_test = "../test.h5"
# 400000 (exactly 403000) top 201497, qcd 201503
# infile_val = "../val.h5"

# max_trks = 200
        all_vals = (
            np.concatenate(arrays)
            if len(arrays) > 0 and arrays[0].size > 0
            else np.array([])
        )
        if all_vals.size == 0:
            pad.Clear()
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
        for cls, arr in zip(classes, arrays):
            name = f"h_{var}_{class_names[cls]}_{idx}"
            h = r.TH1F(name, title, nbins, x_min, x_max)
            h.SetLineColor(colors[cls])
            h.SetLineWidth(2)
            h.SetStats(True)

            for v in arr:
                h.Fill(float(v))

            hists[cls] = h
            KEEP_HISTS.append(h)

        # 描画
        pad.cd()
        first = True
        for j, cls in enumerate(classes):
            opt = "HIST" if first else "HIST SAME"
            first = False
            hists[cls].GetXaxis().SetTitle(xlabel)
            hists[cls].GetYaxis().SetTitle("Entries")
            hists[cls].Draw(opt)
            pad.Update()

            # Stats box をずらす
            st = hists[cls].GetListOfFunctions().FindObject("stats")
            if st:
                st.SetTextColor(colors[cls])
                st.SetLineColor(colors[cls])
                y2 = 0.88 - 0.20 * j
                y1 = y2 - 0.18
                st.SetX1NDC(0.60)
                st.SetX2NDC(0.89)
                st.SetY1NDC(y1)
                st.SetY2NDC(y2)
                pad.Modified()

        # 凡例
        leg = r.TLegend(0.15, 0.75, 0.45, 0.90)
        for cls in classes:
            leg.AddEntry(hists[cls], class_names[cls], "l")
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



#  Main

def main():
    
    print("Loading jets via dump_jets() ...")
    # 必要に応じて dump_jets の引数をここで指定
    for i in range(len(inputs)):
        infile = inputs[i][0]
        ndata = inputs[i][1]
        outfile_top = inputs[i][2]
        outfile_qcd = inputs[i][3]
        print("Input file      :", infile)
        print("# to process    :", ndata)
        print("Output file[top]:", outfile_top)
        print("Output file[qcd]:", outfile_qcd)

        if 1:
            alljets = dump_jets(infile,[0,ndata])

            print(len(alljets['top']),len(alljets['qcd']))

    n_top = len(alljets.get("top", []))
    n_qcd = len(alljets.get("qcd", []))
    print(f"  top jets: {n_top}")
    print(f"  qcd jets: {n_qcd}")

    trk_data, jet_data, M_jet_by_cls, ntrk_jet_by_cls = build_data_from_alljets(alljets)

    out_dir = Path("histograms_root/overlay_top_qcd")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Track 2×3
    plot_track_grid_top_vs_qcd_root(
        trk_data,
        out_dir / "track_overlay_top_qcd_2x3.png",
    )

    # Jet 3×3
    plot_jet_grid_top_vs_qcd_root(
        jet_data,
        M_jet_by_cls,
        ntrk_jet_by_cls,
        out_dir / "jet_overlay_top_qcd_3x3.png",
    )


if __name__ == "__main__":
    main()
