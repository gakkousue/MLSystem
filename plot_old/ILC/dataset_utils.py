#!/usr/bin/env python

from pathlib import Path
import numpy as np

# このファイルの場所からの相対パスで data ディレクトリを指定
DATA_DIRECTORY = (Path(__file__).resolve().parent /
                  "../data/ILC.2019.09-low-level-data").resolve()

# デフォルトで読むファイル
DEFAULT_FNAMES = [
    str(DATA_DIRECTORY / "bb_data1.txt"),   # 0: bb
    str(DATA_DIRECTORY / "cc_data1.txt"),   # 1: cc
    str(DATA_DIRECTORY / "uds_data1.txt"),  # 2: uds
]


def load_dataset(fnames=None, sanity_check=False):
    if fnames is None:
        fnames = DEFAULT_FNAMES

    dat = []
    lab = []
    num = []
    cnt = -1

    for c, fname in enumerate(fnames):
        k = 0
        with open(fname) as fin:
            while True:
                line = fin.readline()
                if not line:
                    break
                if line.rstrip() == "-------------------":
                    k += 1
                    if sanity_check and k == 101:
                        break
                    cnt += 1
                    line = fin.readline()
                    axis = np.array(
                        [float(v) for v in line.rstrip().split()[2:]],
                        dtype=np.float32,
                    )
                    dat.append({"axis": axis, "part": []})
                    lab.append(c)  # 0:bb, 1:cc, 2:uds
                    num.append(0)
                    continue
                part = np.array(
                    [float(v) for v in line.strip().split()],
                    dtype=np.float32,
                )
                dat[cnt]["part"].append(part)

    # 可変長 part を max_part×7 の配列にそろえる
    max_part = 0
    for d in dat:
        if max_part < len(d["part"]):
            max_part = len(d["part"])

    for i in range(len(dat)):
        a = np.zeros((max_part, 7), dtype=np.float32)
        l = len(dat[i]["part"])
        if l == 0:
            l = 1
        else:
            a[:l, :] = np.array(dat[i]["part"])
        dat[i]["part"] = a
        num[i] = l

    return dat, np.array(num, dtype=np.int32), np.array(lab, dtype=np.int32)
