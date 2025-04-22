import numpy as np
import pickle

with open(r'falcon/FALCON/continuous-fair-score-normalization/FALCON/results/train_fmr=0.001/ArcFace/RFW-RFW_VAL/fmr=0.001', "rb") as f:
    a = pickle.load(f)
# a = np.load(r'falcon/FALCON/continuous-fair-score-normalization/FALCON/results/train_fmr=0.001/ArcFace/RFW-RFW_VAL/fmr=0.001', allow_pickle=True)
print(a)