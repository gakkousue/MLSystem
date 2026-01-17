#!/usr/bin/env python

from pathlib import Path
import numpy as np

from dataset_utils import load_dataset as load_dataset_event
from dataset_utils import DEFAULT_FNAMES


def load_dataset_jet(fnames=None, sanity_check=False):
    if fnames is None:
        fnames = DEFAULT_FNAMES

    # まずはイベント単位で読み込む（元の関数を再利用）
    dat_evt, num_evt, lab_evt = load_dataset_event(
        fnames=fnames,
        sanity_check=sanity_check,
    )

    dat_jet_raw = []    # padding前の一時データ
    num_jet_list = []
    lab_jet_list = []

    for i in range(len(dat_evt)):
        axis = dat_evt[i]["axis"]          # たぶん3成分 (ax, ay, az)
        axis3 = axis[:3]                   # 念のため3成分だけ使う
        n_trk = int(num_evt[i])
        parts_full = dat_evt[i]["part"]    # shape: (max_part, 7)

        if n_trk > 0:
            parts = parts_full[:n_trk, :]  # 実トラック数分を切り出し
        else:
            parts = parts_full[:0, :]      # 空 (0×7)

        # トラックの3元運動量 (Px,Py,Pz) を取り出す
        # col: 0=E, 1=Px, 2=Py, 3=Pz
        if n_trk > 0:
            pvec = parts[:, 1:4]  # shape: (n_trk, 3)
            dots = np.dot(pvec, axis3)  # shape: (n_trk,)
            mask_pos = dots >= 0.0
            mask_neg = dots < 0.0

            parts_pos = parts[mask_pos]
            parts_neg = parts[mask_neg]
        else:
            # トラックが0本のイベントの場合：両方空
            parts_pos = parts[:0, :]
            parts_neg = parts[:0, :]

        # 正ジェット (+) : axis はそのまま
        dat_jet_raw.append(
            {
                "axis": axis.copy(),
                "part_list": [row for row in parts_pos],
            }
        )
        num_jet_list.append(len(parts_pos))
        lab_jet_list.append(int(lab_evt[i]))

        # 負ジェット (-) : axis の向きを反転させておくと概念的にわかりやすい
        dat_jet_raw.append(
            {
                "axis": -axis.copy(),
                "part_list": [row for row in parts_neg],
            }
        )
        num_jet_list.append(len(parts_neg))
        lab_jet_list.append(int(lab_evt[i]))

    # ここからは event 版 load_dataset と同様に padding を行う
    max_ntrk_jet = 0
    for d in dat_jet_raw:
        if max_ntrk_jet < len(d["part_list"]):
            max_ntrk_jet = len(d["part_list"])

    dat_jet = []
    for d in dat_jet_raw:
        parts_list = d["part_list"]
        a = np.zeros((max_ntrk_jet, 7), dtype=np.float32)
        l = len(parts_list)
        if l > 0:
            a[:l, :] = np.stack(parts_list).astype(np.float32)
        # l == 0 の場合はゼロのまま
        dat_jet.append(
            {
                "axis": d["axis"].astype(np.float32),
                "part": a,
            }
        )

    num_jet = np.array(num_jet_list, dtype=np.int32)
    lab_jet = np.array(lab_jet_list, dtype=np.int32)
    return dat_jet, num_jet, lab_jet


# 互換性のためのエイリアス（overlay系で load_dataset と書いても使えるように）
def load_dataset(fnames=None, sanity_check=False):
    """イベントではなくジェット単位の load_dataset。"""
    return load_dataset_jet(fnames=fnames, sanity_check=sanity_check)
