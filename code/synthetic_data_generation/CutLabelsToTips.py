import numpy as np
import pandas as pd
import random

dir = ""
dir_source = "Cut/val/labels/"
dir_target = "CutJustTip/val/labels/"

for i in range(0,201):
    for j in range(1,5):
        df = pd.read_csv(dir+dir_source+"Sideview{}_{}.txt".format(i,j), header=None, sep=" ")
        if j < 3:
            idx = df[df[0]==1].index.values.tolist()
            h_new = np.array([random.randint(72,92)*1e-4 for i in range(len(idx))])
            df.loc[idx,2] -= (df.loc[idx,4]-h_new)/2
            df.loc[idx,4] = h_new
        else:
            idx = df[df[0]==1].index.values.tolist()
            h_new = np.array([random.randint(72,92)*1e-4 for i in range(len(idx))])
            df.loc[idx,2] += (df.loc[idx,4]-h_new)/2
            df.loc[idx,4] = h_new
        df.to_csv(dir + dir_target + "Sideview{}_{}.txt".format(i,j), header=None, index=None, sep=" ")