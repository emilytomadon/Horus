import os
import numpy as np

images_list = sorted(os.listdir('/home/gimicaroni/Documents/Datasets/rfw_embeddings/rfw_train'))
IDs = np.array([int(image[2:7]) for image in images_list])
np.save('ids-rfw-train.npy', IDs)