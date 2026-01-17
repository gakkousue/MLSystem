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

inputs = [[infile_train, 10000, "data_top_train_10k.h5", "data_qcd_train_10k.h5"],]

# PyROOT の GC 対策
KEEP_HISTS = []

# 描画スタイル
r.gStyle.SetOptStat(1110)  # entries, mean, RMS
r.gStyle.SetStatBorderSize(1)
r.gStyle.SetStatX(0.88)
r.gStyle.SetStatY(0.88)


#  TLorentzVector を用いたブースト
def boost_trk_and_jet_to_jet_rest(E_trk, px_trk, py_trk, pz_trk,
                                  E_jet, px_jet, py_jet, pz_jet):
    """
    Jet の 4 運動量 (E_jet, p_jet) をもとに、
    - Jet 自身
    - 1 本の Track
    を「Jet 静止系」にブーストした 4-運動量を返す。

    戻り値:
        (E_trk', px_trk', py_trk', pz_trk',
         E_jet', px_jet', py_jet', pz_jet')
    """

    # Jet の4ベクトル（Lab 系）
    jet_lab = r.TLorentzVector()
    jet_lab.SetPxPyPzE(px_jet, py_jet, pz_jet, E_jet)

    # Boost ベクトル（Jet を静止させるには逆方向）
    beta_vec = jet_lab.BoostVector()
    beta_vec *= -1.0

    # Jet を静止系へ
    jet_rest = r.TLorentzVector()
    jet_rest.SetPxPyPzE(px_jet, py_jet, pz_jet, E_jet)
    jet_rest.Boost(beta_vec)

    # Track を静止系へ
    trk_lab = r.TLorentzVector()
    trk_lab.SetPxPyPzE(px_trk, py_trk, pz_trk, E_trk)
    trk_lab.Boost(beta_vec)

    return (
        trk_lab.E(), trk_lab.Px(), trk_lab.Py(), trk_lab.Pz(),
        jet_rest.E(), jet_rest.Px(), jet_rest.Py(), jet_rest.Pz()
    )


#  alljets -> track / jet 物理量の配列に変換
def build_data_from_alljets(alljets):
    """
    alljets(dict) から

      - trk_data[cls]["E_trk"], ...  (Track: Jet 静止系）
      - jet_data[cls]["E_jet"], ...  (Jet: Jet 静止系）
      - ntrk_jet_by_cls[cls]         （ジェット内トラック数）
      - M_jet_by_cls[cls]            （ジェット質量）

    を作る。

    alljets['top'] / alljets['qcd'] は

        jet_info = [1, E_jet, px_jet, py_jet, pz_jet, mass_jet, n_tracks]
        jet_info.extend(track_info)

    というリストのリストと仮定する。
    track_info 部分は (n_tracks, n_feat) をフラットにしたもの。
    1トラックあたりの特徴量数 n_feat = len(track_info) / n_tracks。
    その先頭 4 要素を (E_trk, Px_trk, Py_trk, Pz_trk) と解釈する。
    """

    # Track-level で扱う変数（Jet 静止系）
    trk_vars = ["E_trk", "Px_trk", "Py_trk", "Pz_trk", "P_trk", "Pt_trk"]

    # Jet-level で扱う変数（Jet 静止系
    jet_vars = ["E_jet", "Px_jet", "Py_jet", "Pz_jet", "P_jet", "Pt_jet"]
    classes = ["top", "qcd"]

    trk_data = {cls: {v: [] for v in trk_vars} for cls in classes}
    jet_data = {cls: {v: [] for v in jet_vars} for cls in classes}
    M_jet_by_cls = {cls: [] for cls in classes}
    ntrk_jet_by_cls = {cls: [] for cls in classes}

    for cls in classes:
        jets_cls = alljets.get(cls, [])
        for jet_info in jets_cls:
            if len(jet_info) < 7:
                # ヘッダすら無い異常データはスキップ
                continue

            _, E_jet_lab, px_jet_lab, py_jet_lab, pz_jet_lab, mass_jet, n_tracks = jet_info[:7]
            try:
                n_tracks = int(n_tracks)
            except ValueError:
                continue

            E_jet_lab = float(E_jet_lab)
            px_jet_lab = float(px_jet_lab)
            py_jet_lab = float(py_jet_lab)
            pz_jet_lab = float(pz_jet_lab)
            mass_jet = float(mass_jet)

            # Track 情報（Lab 系での値がフラットに並んでいる）
            track_flat = jet_info[7:]
            if len(track_flat) < n_tracks * 4:
                continue  # 不正なデータ

            # boost
            if n_tracks > 0:
                base0 = 0
                E_trk0_lab = float(track_flat[base0 + 0])
                px_trk0_lab = float(track_flat[base0 + 1])
                py_trk0_lab = float(track_flat[base0 + 2])
                pz_trk0_lab = float(track_flat[base0 + 3])

                (E_trk0_rest, px_trk0_rest, py_trk0_rest, pz_trk0_rest,
                 E_jet_rest, px_jet_rest, py_jet_rest, pz_jet_rest) = boost_trk_and_jet_to_jet_rest(
                    E_trk0_lab, px_trk0_lab, py_trk0_lab, pz_trk0_lab,
                    E_jet_lab, px_jet_lab, py_jet_lab, pz_jet_lab
                )

                # Jet（rest frame）
                P_jet_rest = np.sqrt(px_jet_rest**2 + py_jet_rest**2 + pz_jet_rest**2)
                Pt_jet_rest = np.sqrt(px_jet_rest**2 + py_jet_rest**2)

                jet_data[cls]["E_jet"].append(E_jet_rest)
                jet_data[cls]["Px_jet"].append(px_jet_rest)
                jet_data[cls]["Py_jet"].append(py_jet_rest)
                jet_data[cls]["Pz_jet"].append(pz_jet_rest)
                jet_data[cls]["P_jet"].append(P_jet_rest)
                jet_data[cls]["Pt_jet"].append(Pt_jet_rest)

                M_jet_by_cls[cls].append(mass_jet)
                ntrk_jet_by_cls[cls].append(n_tracks)

                # 最初のトラックも含めて、全トラックを Jet 静止系へ
                # 1本目トラック
                P_trk0_rest = np.sqrt(px_trk0_rest**2 + py_trk0_rest**2 + pz_trk0_rest**2)
                Pt_trk0_rest = np.sqrt(px_trk0_rest**2 + py_trk0_rest**2)

                trk_data[cls]["E_trk"].append(E_trk0_rest)
                trk_data[cls]["Px_trk"].append(px_trk0_rest)
                trk_data[cls]["Py_trk"].append(py_trk0_rest)
                trk_data[cls]["Pz_trk"].append(pz_trk0_rest)
                trk_data[cls]["P_trk"].append(P_trk0_rest)
                trk_data[cls]["Pt_trk"].append(Pt_trk0_rest)

                # 2本目以降
                for i_trk in range(1, n_tracks):
                    base = i_trk * 4
                    try:
                        E_trk_lab = float(track_flat[base + 0])
                        px_trk_lab = float(track_flat[base + 1])
                        py_trk_lab = float(track_flat[base + 2])
                        pz_trk_lab = float(track_flat[base + 3])
                    except IndexError:
                        break

                    # 同じ boost ベクトルを共有したいところだが、
                    # 実装簡単化のため毎回 boost 関数を呼び出す。
                    (E_trk_rest, px_trk_rest, py_trk_rest, pz_trk_rest,
                     _, _, _, _) = boost_trk_and_jet_to_jet_rest(
                        E_trk_lab, px_trk_lab, py_trk_lab, pz_trk_lab,
                        E_jet_lab, px_jet_lab, py_jet_lab, pz_jet_lab
                    )

                    P_trk_rest = np.sqrt(px_trk_rest**2 + py_trk_rest**2 + pz_trk_rest**2)
                    Pt_trk_rest = np.sqrt(px_trk_rest**2 + py_trk_rest**2)

                    trk_data[cls]["E_trk"].append(E_trk_rest)
                    trk_data[cls]["Px_trk"].append(px_trk_rest)
                    trk_data[cls]["Py_trk"].append(py_trk_rest)
                    trk_data[cls]["Pz_trk"].append(pz_trk_rest)
                    trk_data[cls]["P_trk"].append(P_trk_rest)
                    trk_data[cls]["Pt_trk"].append(Pt_trk_rest)

    return trk_data, jet_data, M_jet_by_cls, ntrk_jet_by_cls


#  Track-level 2×3 overlay
# -------------------------
#  共通：軸範囲をパーセンタイルで決める（※残しておく）
# -------------------------
def get_axis_range_percentile(arrays, p_lo=0.5, p_hi=99.5):
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


# ============================================================
#  ここから「軸設定だけを辞書で管理」するための追加・置換部分
#  （boost後プロットでも同じ方式）
# ============================================================

AXIS_CFG_TRACK_BOOST = {
    # まずは無難な例（必要に応じて値をあなたが調整）
    "E_trk":  {"nbins": 100, "xmin": 0.0,    "xmax": 200.0, "logy": False, "ymin": 0.0, "ymax": None, "ymax_factor": 1.25},
    "Px_trk": {"nbins": 120, "xmin": -60.0,  "xmax": 60.0,  "logy": False, "ymin": 0.0, "ymax": None, "ymax_factor": 1.25},
    "Py_trk": {"nbins": 120, "xmin": -60.0,  "xmax": 60.0,  "logy": False, "ymin": 0.0, "ymax": None, "ymax_factor": 1.25},
    "Pz_trk": {"nbins": 160, "xmin": -80.0,  "xmax": 80.0,  "logy": False, "ymin": 0.0, "ymax": None, "ymax_factor": 1.25},
    "P_trk":  {"nbins": 120, "xmin": 0.0,    "xmax": 120.0, "logy": False, "ymin": 0.0, "ymax": None, "ymax_factor": 1.25},
    "Pt_trk": {"nbins": 120, "xmin": 0.0,    "xmax": 80.0,  "logy": False, "ymin": 0.0, "ymax": None, "ymax_factor": 1.25},
}

AXIS_CFG_JET_BOOST = {
    # Jet rest frame なので、理想的には Px,Py,Pz は 0 付近に集まる
    "E_jet":  {"nbins": 100, "xmin": 0.0,    "xmax": 500.0, "logy": False, "ymin": 0.0, "ymax": None, "ymax_factor": 1.25},
    "Px_jet": {"nbins": 120, "xmin": -5.0,   "xmax": 5.0,   "logy": False, "ymin": 0.0, "ymax": None, "ymax_factor": 1.25},
    "Py_jet": {"nbins": 120, "xmin": -5.0,   "xmax": 5.0,   "logy": False, "ymin": 0.0, "ymax": None, "ymax_factor": 1.25},
    "Pz_jet": {"nbins": 120, "xmin": -5.0,   "xmax": 5.0,   "logy": False, "ymin": 0.0, "ymax": None, "ymax_factor": 1.25},
    "P_jet":  {"nbins": 120, "xmin": 0.0,    "xmax": 10.0,  "logy": False, "ymin": 0.0, "ymax": None, "ymax_factor": 1.25},
    "Pt_jet": {"nbins": 120, "xmin": 0.0,    "xmax": 10.0,  "logy": False, "ymin": 0.0, "ymax": None, "ymax_factor": 1.25},

    "M_jet":  {"nbins": 120, "xmin": 0.0,    "xmax": 250.0, "logy": False, "ymin": 0.0, "ymax": None, "ymax_factor": 1.25},

    # trk_jet は整数ビン自動
    "trk_jet": {"auto_integer_bins": True, "logy": False, "ymin": 0.0, "ymax": None, "ymax_factor": 1.25},
}


def _get_hist_maximum(hists, include_error=True):
    ymax = 0.0
    for h in hists:
        if not h:
            continue
        m = h.GetMaximum()
        if include_error:
            ib = h.GetMaximumBin()
            m = max(m, h.GetBinContent(ib) + h.GetBinError(ib))
        if m > ymax:
            ymax = m
    return float(ymax)


def _apply_pad_axis_settings(pad, hists_dict, xlabel, ytitle, cfg):
    logy = bool(cfg.get("logy", False))
    pad.SetLogy(1 if logy else 0)

    ymin = cfg.get("ymin", None)
    ymax = cfg.get("ymax", None)
    hlist = list(hists_dict.values())

    if ymax is None:
        auto_max = _get_hist_maximum(hlist, include_error=True)
        fac = float(cfg.get("ymax_factor", 1.25))
        ymax = auto_max * fac if auto_max > 0 else 1.0

    if ymin is None:
        ymin = float(cfg.get("ymin_for_log", 0.5)) if logy else 0.0

    # top（最初）を代表として軸設定
    h0 = hlist[0]
    h0.GetXaxis().SetTitle(xlabel)
    h0.GetYaxis().SetTitle(ytitle)
    h0.SetMinimum(float(ymin))
    h0.SetMaximum(float(ymax))


# -----------------------------------------
#  Track-level 2×3 overlay（Jet 静止系） ※置換
# -----------------------------------------
def plot_track_grid_top_vs_qcd_root(trk_data, out_path):
    """
    trk_data[cls][var] から、top / qcd の Track-level ヒストグラムを
    2×3 キャンバスで overlay 描画する（Jet 静止系）。

    軸設定は AXIS_CFG_TRACK_BOOST（辞書）で管理する。
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
    colors = {"top": r.kRed + 1, "qcd": r.kBlue + 1}

    c = r.TCanvas("c_tracks_topqcd", "Top vs QCD tracks (jet rest)", 1800, 1000)
    c.Divide(3, 2)

    for idx, (var, xlabel, title) in enumerate(plot_defs):
        pad = c.cd(idx + 1)

        cfg = AXIS_CFG_TRACK_BOOST[var]
        nbins = int(cfg["nbins"])
        x_min = float(cfg["xmin"])
        x_max = float(cfg["xmax"])

        hists = {}
        for cls in classes:
            arr = np.array(trk_data[cls][var], dtype=np.float64)

            name = f"h_{var}_{class_names[cls]}_{idx}"
            h = r.TH1F(name, title, nbins, x_min, x_max)
            h.SetLineColor(colors[cls])
            h.SetLineWidth(2)
            h.SetStats(True)

            for v in arr:
                h.Fill(float(v))

            hists[cls] = h
            KEEP_HISTS.append(h)

        _apply_pad_axis_settings(pad, hists, xlabel, "Entries", cfg)

        pad.cd()
        first = True
        for j, cls in enumerate(classes):
            opt = "HIST" if first else "HIST SAME"
            first = False
            hists[cls].Draw(opt)
            pad.Update()

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


# -----------------------------------------
#  Jet-level 3×3 overlay（Jet 静止系） ※置換
# -----------------------------------------
def plot_jet_grid_top_vs_qcd_root(jet_data, M_jet_by_cls, ntrk_jet_by_cls, out_path):
    """
    jet_data / M_jet_by_cls / ntrk_jet_by_cls から、
    top / qcd の Jet-level ヒストグラムを 3×3 キャンバスで overlay 描画する。
    （ここでは Jet 自身も Jet 静止系の 4-運動量）

    軸設定は AXIS_CFG_JET_BOOST（辞書）で管理する。
    """

    plot_defs = [
        ("E_jet",   "E_jet [GeV]",   "Jet energy E_jet (jet rest)",               "jet"),
        ("Px_jet",  "Px_jet [GeV]",  "Jet momentum Px_jet (jet rest)",            "jet"),
        ("Py_jet",  "Py_jet [GeV]",  "Jet momentum Py_jet (jet rest)",            "jet"),
        ("Pz_jet",  "Pz_jet [GeV]",  "Jet momentum Pz_jet (jet rest)",            "jet"),
        ("P_jet",   "P_jet [GeV]",   "Jet momentum |P_jet| (jet rest)",           "jet"),
        ("Pt_jet",  "Pt_jet [GeV]",  "Jet transverse momentum Pt_jet (jet rest)", "jet"),
        ("M_jet",   "M_jet [GeV]",   "Jet mass M_jet",                            "mass"),
        ("trk_jet", "trk_jet",       "Tracks per jet (trk_jet)",                  "ntrk"),
    ]

    classes = ["top", "qcd"]
    class_names = {"top": "top", "qcd": "qcd"}
    colors = {"top": r.kRed + 1, "qcd": r.kBlue + 1}

    c = r.TCanvas("c_jets_topqcd", "Top vs QCD jets (jet rest frame)", 1800, 1200)
    c.Divide(3, 3)

    # trk_jet 用（auto_integer_bins のときに使う）
    all_ntrk = []
    for cls in classes:
        all_ntrk.extend(ntrk_jet_by_cls[cls])
    all_ntrk = np.array(all_ntrk, dtype=np.int32)
    if all_ntrk.size > 0:
        ntrk_min = int(all_ntrk.min())
        ntrk_max = int(all_ntrk.max())
    else:
        ntrk_min = 0
        ntrk_max = 0

    for idx, (var, xlabel, title, kind) in enumerate(plot_defs):
        pad = c.cd(idx + 1)

        cfg = AXIS_CFG_JET_BOOST[var]

        if kind == "jet":
            arrays = {cls: np.array(jet_data[cls][var], dtype=np.float64) for cls in classes}
        elif kind == "mass":
            arrays = {cls: np.array(M_jet_by_cls[cls], dtype=np.float64) for cls in classes}
        elif kind == "ntrk":
            arrays = {cls: np.array(ntrk_jet_by_cls[cls], dtype=np.float64) for cls in classes}
        else:
            pad.Clear()
            continue

        # xビン設定（辞書から：手動 or 整数ビン自動）
        if kind == "ntrk" and bool(cfg.get("auto_integer_bins", False)):
            nbins = ntrk_max - ntrk_min + 1
            x_min = ntrk_min - 0.5
            x_max = ntrk_max + 0.5
        else:
            nbins = int(cfg["nbins"])
            x_min = float(cfg["xmin"])
            x_max = float(cfg["xmax"])

        hists = {}
        for cls in classes:
            name = f"h_{var}_{class_names[cls]}_{idx}"
            h = r.TH1F(name, title, nbins, x_min, x_max)
            h.SetLineColor(colors[cls])
            h.SetLineWidth(2)
            h.SetStats(True)

            for v in arrays[cls]:
                h.Fill(float(v))

            hists[cls] = h
            KEEP_HISTS.append(h)

        _apply_pad_axis_settings(pad, hists, xlabel, "Entries", cfg)

        pad.cd()
        first = True
        for j, cls in enumerate(classes):
            opt = "HIST" if first else "HIST SAME"
            first = False
            hists[cls].Draw(opt)
            pad.Update()

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
    for i in range(len(inputs)):
        infile = inputs[i][0]
        ndata = inputs[i][1]
        outfile_top = inputs[i][2]
        outfile_qcd = inputs[i][3]
        print("Input file      :", infile)
        print("# to process    :", ndata)
        print("Output file[top]:", outfile_top)
        print("Output file[qcd]:", outfile_qcd)

        alljets = dump_jets(infile, [0, ndata])
        print(len(alljets['top']), len(alljets['qcd']))

    n_top = len(alljets.get("top", []))
    n_qcd = len(alljets.get("qcd", []))
    print(f"  top jets: {n_top}")
    print(f"  qcd jets: {n_qcd}")

    trk_data, jet_data, M_jet_by_cls, ntrk_jet_by_cls = build_data_from_alljets(alljets)

    out_dir = Path("histograms_root/overlay_top_qcd_boost_setting")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Track 2×3（Jet 静止系）
    plot_track_grid_top_vs_qcd_root(
        trk_data,
        out_dir / "track_overlay_top_qcd_boost_2x3.png",
    )

    # Jet 3×3（Jet 静止系）
    plot_jet_grid_top_vs_qcd_root(
        jet_data,
        M_jet_by_cls,
        ntrk_jet_by_cls,
        out_dir / "jet_overlay_top_qcd_boost_3x3.png",
    )


if __name__ == "__main__":
    main()
