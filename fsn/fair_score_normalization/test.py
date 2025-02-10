import numpy as np

a = np.load('ids-rfw-train.npy')
b = np.load('cv-mask-rfw.npy')
print(a.shape, b.shape)