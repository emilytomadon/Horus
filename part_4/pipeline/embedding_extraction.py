import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from tqdm import tqdm

SIMILARITY_THRESHOLD = 0.5
DATASET_PATH = r'/home/gimicaroni/Documents/Datasets/Umdfaces_split/train'
RESULT_PATH = r'/home/gimicaroni/Documents/Datasets/Umdfaces_split_embeddings/train'

def get_embedding(image_path): # Se quiser fazer com o deepface, é só mudar essa função
    img = cv2.imread(image_path)
    faces = app.get(img)
    if len(faces) > 0:
        return faces[0].embedding  # Returns the embedding of the first detected face
    return None

def extract_all_embeddings():
    os.makedirs(RESULT_PATH, exist_ok=True)
    all_embeddings_dict = {}

    for root, dirs, files in tqdm(os.walk(DATASET_PATH)):

        for dir_name in tqdm(sorted(dirs)):
            dir_path = os.path.join(DATASET_PATH, dir_name)
            result_dir_path = os.path.join(RESULT_PATH, dir_name)
            os.makedirs(result_dir_path, exist_ok=True)

            for filename in sorted(os.listdir(dir_path)):
                file_path = os.path.join(dir_path, filename)

                if file_path.endswith(('.png', '.jpg', '.jpeg')):
                    embedding_path = os.path.join(result_dir_path, filename).split('.')[0] #corta o 00001/00001.jpg no ponto e pega o que vem antes
                    embedding = get_embedding(file_path)
                    all_embeddings_dict[filename] = embedding
                    np.save(f'{embedding_path}.npy', embedding)

    np.savez_compressed(f'{RESULT_PATH}/embeddings_arcface_dict.npz', **all_embeddings_dict) #salva um dicionário com todos os embeddings também, além de cada embedding separadamente

if __name__ == "__main__":
    app = FaceAnalysis(name='buffalo_l')  # 'buffalo_l' includes ArcFace iResNet100
    app.prepare(ctx_id=0, det_thresh=0.5)  # ctx_id=0 for CPU, ctx_id=1 for GPU
    
    extract_all_embeddings()
    print("Extraction of embeddings complete.")
    
