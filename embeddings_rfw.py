import os
import numpy as np

IMAGES_DIR = '/home/gimicaroni/Documents/Datasets/rfw_embeddings/rfw_train'

images_list = sorted(os.listdir(IMAGES_DIR))
array = np.load(os.path.join(IMAGES_DIR, images_list[0]))
for image in images_list[1:]:
    print(image)
    img_path = os.path.join(IMAGES_DIR, image)
    current_img = np.load(img_path)
    array = np.vstack((array, current_img))
np.save('embeddings-array-rfw.npy', array)