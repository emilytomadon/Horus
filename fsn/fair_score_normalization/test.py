import numpy as np

a = np.load('ids-rfw-train.npy')
b = np.load('cv-mask-rfw.npy')
c = np.load('embeddings-array-rfw.npy')
print(a.shape, b.shape, c.shape)
print(c)
