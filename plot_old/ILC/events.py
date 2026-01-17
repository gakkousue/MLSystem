import pickle
import numpy as np
from train_pytorch import RCNPDataset

path = "../data/ILC.2019.09-low-level-data/"


for split in ["train", "valid", "test"]:
    ds = pickle.load(open(path + f"rcnpdataset_{split}.pickle", "rb"))
    labs = np.array(ds.lab)
    print(f"{split}: total={len(labs)}, bb={np.sum(labs==0)}, cc={np.sum(labs==1)}, uds={np.sum(labs==2)}")
