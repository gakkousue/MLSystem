# definitions/datasets/rcnp/plots.py
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from MLsystem.utils.base_plot import BasePlot


class InputDataStatistics(BasePlot):
    name = "Input Data Statistics"
    description = "データセット(キャッシュ)の物理量分布をクラスごとに表示します"

    def execute(self):
        # 1. Datamoduleのセットアップ (全データが読み込まれる)
        _, datamodule = self.loader.setup(stage="fit")

        print(f"[{self.name}] Combining all dataset splits (Train/Val/Test)...")

        # 2. Train, Val, Test 各データセットからTensorを取り出して結合する
        datasets = [datamodule.train_ds, datamodule.val_ds, datamodule.test_ds]
        
        labels = np.concatenate([ds.label.numpy() for ds in datasets])
        parts = np.concatenate([ds.part.numpy() for ds in datasets])
        nums = np.concatenate([ds.num.numpy() for ds in datasets])

        data_dict = {
            "label": labels,
            "part": parts,
            "num": nums
        }

        IDX_PX, IDX_PY, IDX_PZ, IDX_E = 0, 1, 2, 3

        # labels = data_dict["label"]  # (N,)
        # parts = data_dict["part"]    # (N, P, 7)
        # nums = data_dict["num"]      # (N,)

        class_names = getattr(datamodule, "class_names", ["bb", "cc", "uds"])
        # overlay_hists.py に合わせた配色順
        colors = ["blue", "orange", "green"]

        print(f"[{self.name}] Calculating physics quantities...")

        # stats辞書の初期化 (Px, Py, Pz を追加)
        stats = {
            cls_idx: {
                "n_trk": [],
                "trk_pt": [], "trk_p": [], "trk_e": [],
                "trk_px": [], "trk_py": [], "trk_pz": [],
                "jet_pt": [], "jet_p": [], "jet_e": [], "jet_m": [],
                "jet_px": [], "jet_py": [], "jet_pz": []
            }
            for cls_idx in range(len(class_names))
        }

        # クラスごとに集計
        for cls_idx in range(len(class_names)):
            mask = (labels == cls_idx)
            parts_cls = parts[mask]  # (N_cls, P, 7)
            nums_cls = nums[mask]    # (N_cls,)

            # トラック数
            stats[cls_idx]["n_trk"] = nums_cls

            # --- Track Level ---
            # パディング(E=0)を除外してフラット化
            parts_flat = parts_cls.reshape(-1, 7)
            valid_mask = (parts_flat[:, IDX_E] != 0)
            parts_valid = parts_flat[valid_mask]

            px = parts_valid[:, IDX_PX]
            py = parts_valid[:, IDX_PY]
            pz = parts_valid[:, IDX_PZ]
            e = parts_valid[:, IDX_E]

            stats[cls_idx]["trk_e"] = e
            stats[cls_idx]["trk_px"] = px
            stats[cls_idx]["trk_py"] = py
            stats[cls_idx]["trk_pz"] = pz
            stats[cls_idx]["trk_p"] = np.sqrt(px**2 + py**2 + pz**2)
            stats[cls_idx]["trk_pt"] = np.sqrt(px**2 + py**2)

            # --- Jet Level ---
            # 粒子ごとに和をとってジェットの物理量を計算
            jet_px = np.sum(parts_cls[:, :, IDX_PX], axis=1)
            jet_py = np.sum(parts_cls[:, :, IDX_PY], axis=1)
            jet_pz = np.sum(parts_cls[:, :, IDX_PZ], axis=1)
            jet_e = np.sum(parts_cls[:, :, IDX_E], axis=1)

            jet_pt = np.sqrt(jet_px**2 + jet_py**2)
            jet_p = np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2)

            jet_m = np.sqrt(jet_e**2 - jet_p**2)

            stats[cls_idx]["jet_e"] = jet_e
            stats[cls_idx]["jet_px"] = jet_px
            stats[cls_idx]["jet_py"] = jet_py
            stats[cls_idx]["jet_pz"] = jet_pz
            stats[cls_idx]["jet_p"] = jet_p
            stats[cls_idx]["jet_pt"] = jet_pt
            stats[cls_idx]["jet_m"] = jet_m

        # --- Plotting ---
        # 1. Track Level 2x3 Grid (overlay_hists.py 相当)
        self._plot_track_grid_2x3(stats, class_names, colors, "input_stats_track_2x3.png")

        # 2. Jet Level 3x3 Grid (overlay_hists.py 相当)
        self._plot_jet_grid_3x3(stats, class_names, colors, "input_stats_jet_3x3.png")

        # 3. Track vs Jet Pair Grid 3x2 (plot_histograms23.py 相当)
        self._plot_pair_grid_3x2(stats, class_names, colors, "input_stats_pair_3x2.png")

    def _plot_track_grid_2x3(self, stats, class_names, colors, filename):
        """overlay_hists.py の plot_track_grid 相当"""
        plot_defs = [
            ("trk_e",  "E_trk [GeV]",  "Track energy E_trk"),
            ("trk_px", "Px_trk [GeV]", "Track momentum Px_trk"),
            ("trk_py", "Py_trk [GeV]", "Track momentum Py_trk"),
            ("trk_pz", "Pz_trk [GeV]", "Track momentum Pz_trk"),
            ("trk_p",  "P_trk [GeV]",  "Track momentum |P_trk|"),
            ("trk_pt", "Pt_trk [GeV]", "Track transverse momentum Pt_trk"),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for idx, (key, xlabel, title) in enumerate(plot_defs):
            r, c = divmod(idx, 3)
            ax = axes[r, c]

            for cls_idx, name in enumerate(class_names):
                ax.hist(
                    stats[cls_idx][key], bins=100, histtype="step",
                    color=colors[cls_idx % len(colors)], label=name, linewidth=1.2
                )
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Entries")
            ax.grid(True, which="both", linestyle="--", alpha=0.5)

        axes[0, 0].legend()
        plt.tight_layout()
        save_path = os.path.join(self.loader.exp_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"[{self.name}] Saved: {save_path}")

    def _plot_jet_grid_3x3(self, stats, class_names, colors, filename):
        """overlay_hists.py の plot_jet_grid 相当"""
        plot_defs = [
            ("jet_e",   "E_jet [GeV]",   "Jet energy E_jet"),
            ("jet_px",  "Px_jet [GeV]",  "Jet momentum Px_jet"),
            ("jet_py",  "Py_jet [GeV]",  "Jet momentum Py_jet"),
            ("jet_pz",  "Pz_jet [GeV]",  "Jet momentum Pz_jet"),
            ("jet_p",   "P_jet [GeV]",   "Jet momentum |P_jet|"),
            ("jet_pt",  "Pt_jet [GeV]",  "Jet transverse momentum Pt_jet"),
            ("jet_m",   "M_jet [GeV]",   "Jet mass M_jet"),
            ("n_trk",   "trk_jet",       "Tracks per jet (trk_jet)"),
        ]

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        # n_trk用の共通ビン作成
        all_ntrk = np.concatenate([stats[i]["n_trk"] for i in range(len(class_names))])
        ntrk_min, ntrk_max = int(all_ntrk.min()), int(all_ntrk.max())
        ntrk_bins = np.arange(ntrk_min - 0.5, ntrk_max + 1.5, 1.0)

        for idx, (key, xlabel, title) in enumerate(plot_defs):
            r, c = divmod(idx, 3)
            ax = axes[r, c]

            bins = ntrk_bins if key == "n_trk" else 100

            for cls_idx, name in enumerate(class_names):
                ax.hist(
                    stats[cls_idx][key], bins=bins, histtype="step",
                    color=colors[cls_idx % len(colors)], label=name, linewidth=1.2
                )
                
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Entries")
            ax.grid(True, which="both", linestyle="--", alpha=0.5)

        # 残りのパネルをオフ
        for idx in range(len(plot_defs), 9):
            r, c = divmod(idx, 3)
            axes[r, c].axis("off")

        axes[0, 0].legend()
        plt.tight_layout()
        save_path = os.path.join(self.loader.exp_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"[{self.name}] Saved: {save_path}")

    def _plot_pair_grid_3x2(self, stats, class_names, colors, filename):
        """plot_histograms23.py のペアプロット相当 (左:Track, 右:Jet)"""
        # (Track Key, Jet Key, Title Prefix)
        pairs = [
            ("n_trk", "jet_m",  "trk_jet vs M_jet"),
            ("trk_e", "jet_e",   "Energy"),
            ("trk_px", "jet_px", "Px"),
            ("trk_py", "jet_py", "Py"),
            ("trk_pz", "jet_pz", "Pz"),
            ("trk_p", "jet_p",   "Momentum |P|"),
            ("trk_pt", "jet_pt", "Transverse Momentum Pt"),
        ]

        # 7ペアあるので、今回は plot_histograms23.py と同じく 3x2 の図を複数枚...ではなく
        # 全部を1つの大きな図にするか、主要なものを並べるか。
        # plot_histograms23.py は 7枚の画像を出力しているが、ここでは 4x2 のグリッドにまとめて1枚に出力する形に統合する。
        
        rows = 4
        cols = 2
        fig, axes = plt.subplots(rows, cols, figsize=(12, 16))
        
        for idx, (k1, k2, title_pfx) in enumerate(pairs):
            if idx >= rows * 1: # 左列を使う想定だが、ペアを左右に並べるなら行ごとに処理
                pass

            # 左: Track / 右: Jet
            ax_l = axes[idx % rows, 0] # 左列には入らないのでレイアウト調整が必要
            # plot_histograms23.py は (Track vs Jet) を 1つのペアとして 3x2 の図を作っているわけではなく
            # "3x2の図の中に、左列Track、右列Jet" という構成を 7パターン作っている。
            # ここではシンプルに「左列=Track, 右列=Jet」として、行ごとに変数を変える 7行2列...は長すぎる。
            # 今回は提供されたコードの `plot_3x2_pair` の精神（左右比較）を再現するため、
            # 代表的なペアを 1枚の大きな図 (例: 4行2列) に収めるか、ループで複数枚出すのが適切だが、
            # execute内で完結させるため、ここでは「主要な変数」をピックアップして 4行2列 (計8枠=4ペア) で出力する。
            # 優先度: n_trk/M_jet, E, Pt, P
            pass
        
        # 方針変更: plot_histograms23.py を再現し、全ペアを含む長い図 (7行2列) を作成する
        fig, axes = plt.subplots(7, 2, figsize=(10, 20))
        
        for i, (k1, k2, title_pfx) in enumerate(pairs):
            # 左: Track
            ax_l = axes[i, 0]
            for cls_idx, name in enumerate(class_names):
                ax_l.hist(stats[cls_idx][k1], bins=50 if k1=="n_trk" else 100, 
                          histtype="step", color=colors[cls_idx % len(colors)], label=name)
            ax_l.set_title(f"Track: {title_pfx}")
            ax_l.grid(True, alpha=0.3)

            # 右: Jet
            ax_r = axes[i, 1]
            for cls_idx, name in enumerate(class_names):
                ax_r.hist(stats[cls_idx][k2], bins=100, 
                          histtype="step", color=colors[cls_idx % len(colors)], label=name)
            ax_r.set_title(f"Jet: {title_pfx}")
            ax_r.grid(True, alpha=0.3)
            
            if i == 0: ax_l.legend()

        plt.tight_layout()
        save_path = os.path.join(self.loader.exp_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"[{self.name}] Saved: {save_path}")