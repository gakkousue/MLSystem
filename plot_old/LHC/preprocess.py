import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyjet import DTYPE_PTEPM

#
# This "preprocess.py" is a program (preprocess) to store data pd.DataFrame(hdf) from h5
#

# Dataset: TopQuark jets
# https://zenodo.org/record/2603256#.YhM3YZZUtmM
# /data/multiai/data2/TopQuarkRefData/README
# In total 1.2M training events, 400k validation events and 400k test events.
#
# 1200000 (exactly 1211000) top 605477, qcd 605523
infile_train = "/data/multiai/data2/TopQuarkRefData/data/train.h5"
# 400000 (exactly 404000) top 202086, qcd 201914
infile_test = "/data/multiai/data2/TopQuarkRefData/data/test.h5"
# 400000 (exactly 403000) top 201497, qcd 201503
infile_val = "/data/multiai/data2/TopQuarkRefData/data/val.h5"

max_trks = 200
    
# orignal input file, # of events to proceed, file names for top and qcd (respectively)
inputs = [ 
#    [infile_train, 1211000, "data_top_train_1211k.h5", "data_qcd_train_1211k.h5"],
#    [infile_train,  500000, "data_top_train_500k.h5",  "data_qcd_train_500k.h5" ],
#    [infile_train,  100000, "data_top_train_100k.h5",  "data_qcd_train_100k.h5" ],
#    [infile_train,   50000, "data_top_train_50k.h5",   "data_qcd_train_50k.h5"  ],
    [infile_train,   10000, "data_top_train_10k.h5",   "data_qcd_train_10k.h5"  ],
#    [infile_val,    403000, "data_top_val_403k.h5",    "data_qcd_val_403k.h5"   ],
#    [infile_val,    100000, "data_top_val_100k.h5",    "data_qcd_val_100k.h5"   ],
#    [infile_test,   404000, "data_top_test_404k.h5",   "data_qcd_test_404k.h5"  ],
#    [infile_test,   100000, "data_top_test_100k.h5",   "data_qcd_test_100k.h5"  ], 
]

def dump_jets(infile, evtrange):
    # pandas
    df = pd.read_hdf(infile, start=evtrange[0], stop=evtrange[1], key="table")

    # [ njet=1, [(E,px,py,pz)_jet, mass_jet, ntrk, (E.px,py,pz)_trk1, ...] ]

    alljets = {} # dict
    alljets['top'] = [] # list
    alljets['qcd'] = []
    for jet in df.iterrows():
        # this "jet" has two components: jet[0] and jet[1].
        #print(len(jet)) # = 2
        #print("INIT0:",jet[0]) # as Name (just index by using integer)
        #print("INIT1:",jet[1]) # tracks information for each jet (main data)
        #print("INIT1:",jet[1].shape) # (806,)
        # 806: (0-199)x4[E,PX,PY,PZ] for tracks, 4 for truth top[E,PX,PY,PZ], the last two is for ttv and is_signal_new.

        jet = jet[1]

        # "::4" is "index with 4-step"
        tracks = np.zeros(len([0 for i in jet[:800:4] if i!=0]), dtype=DTYPE_PTEPM)
        n_tracks = len(tracks)

        # jet init
        E_jet = px_jet = py_jet = pz_jet = 0.
        
        # tracks
        track_info = []
        for j in range(n_tracks): # pt ordering
            E = jet[j*4+0]
            px = jet[j*4+1]
            py = jet[j*4+2]
            pz = jet[j*4+3]
            #print("j,pt,E,p=",j,np.sqrt(px*px+py*py),E,np.sqrt(px*px+py*py+pz*pz))
            track_info.extend([E,px,py,pz])
            E_jet += E
            px_jet += px
            py_jet += py
            pz_jet += pz

        # jet
        mass_jet = E_jet*E_jet-(px_jet*px_jet+py_jet*py_jet+pz_jet*pz_jet)
        mass_jet = np.sign(mass_jet)*np.sqrt(np.abs(mass_jet))
        jet_info = [1,E_jet,px_jet,py_jet,pz_jet,mass_jet,n_tracks]
        #print(jet_info)

        jet_info.extend(track_info)
        if n_tracks < max_trks:
            jet_info.extend([0]*((max_trks-n_tracks)*4))
        #
        if jet['is_signal_new'] == True:
            alljets['top'] += [jet_info]
        else:
            alljets['qcd'] += [jet_info]

    return alljets

def store_data2(data, filename):
    start_index = 0
    batch_size = 128*8
    n_events = len(data)

    #print(n_events)
    
    while start_index < n_events:
        end_index = min(start_index+batch_size,n_events)
        #print(start_index,end_index)
        df = data[start_index:end_index]
        event = []
        for i in range(len(df)):
            #print("df's length:",len(df),df[0])
            dnp=np.array(df[i])
            event += [dnp]
            #print("idx=",i,event)
            
        df_out = pd.DataFrame(data=event,
                              index=np.arange(start_index,end_index))

        #print(df_out)
        
        df_out.to_hdf(filename,
                      key='table',
                      append=(start_index!=0),
                      format='table',
                      complib='blosc',
                      complevel=5)

        start_index += batch_size

if __name__ == "__main__":
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

            # store or not
            if 1:
                store_data2(alljets['top'],outfile_top)
                store_data2(alljets['qcd'],outfile_qcd)
