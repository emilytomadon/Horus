import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def load_embeddings_from_folder(path):
    embeddings = []
    filenames = []
    try:

        for folder in tqdm(sorted(os.listdir(path))):
            folder_path = os.path.join(path, folder)
            for filename in sorted(os.listdir(folder_path)):
                if filename.lower().endswith((".jpg", ".jpeg", ".png", ".csv")):
                    embedding_path = os.path.join(folder_path, filename)
                    embedding_csv = pd.read_csv(embedding_path, header=None)
                    embedding_array = np.array(embedding_csv)
                    # print(embedding_array.shape)
                    embeddings.append(embedding_array)
                    index_id = embedding_path.find("train/") + 6
                    identity = embedding_path[index_id:-4]
                    filenames.append(embedding_path[index_id:-4])
                    # print(f'Embedding {embedding_path[index_id:-4]} loaded')
        return embeddings, filenames
    except KeyboardInterrupt:
        return embeddings, filenames


if __name__ == "__main__":
    embeddings_path = r'/home/gimicaroni/Documents/Datasets/UMDFaces_griaule/umdfaces_features/csv/train'
    try:
        embeddings_list, filenames = load_embeddings_from_folder(embeddings_path)
    finally:
        embeddings = np.array(embeddings_list)
        embeddings = embeddings.squeeze(axis=1)
        np.save('embeddings_array_umd_griaule_train.npy', embeddings)

        df = pd.DataFrame(filenames)
        embedding_df = pd.DataFrame(embeddings, columns=[f'embedding_{i}' for i in range(embeddings.shape[1])])

        result = pd.concat([df, embedding_df], axis=1)

        print(result.head)
        result.to_csv('umdfaces_embeddings_griaule_train2.csv')
        # np.save("embeddings_umdfaces_griaule_train.npy", np.array(embeddings_list))
