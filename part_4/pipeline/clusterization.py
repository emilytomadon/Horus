from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm

EMBEDDINGS_DIR = r'/home/gimicaroni/Documents/Datasets/Umdfaces_split_embeddings/train'
N_CLUSTERS = 15

def build_embedding_dict():
    embeddings_dict = dict()
    for root, dirs, files in os.walk(EMBEDDINGS_DIR):
        for dir_name in tqdm(sorted(dirs)):
            dir_path = os.path.join(EMBEDDINGS_DIR, dir_name)
            for filename in sorted(os.listdir(dir_path)):
                if filename.endswith(('.npy')):
                    embedding_path = os.path.join(dir_path, filename)
                    embeddings_dict[filename] = np.load(embedding_path, allow_pickle=True)
                    print(embeddings_dict[filename])
    return embeddings_dict

def preprocess(): #carrega o dicionario de embeddings, normaliza ele, e devolve o dicionário, uma lista só dos ids e uma matriz só com os embeddings
    embeddings_dict = dict(np.load(os.path.join(EMBEDDINGS_DIR, 'embeddings_arcface_dict.npz'), allow_pickle=True))
    processed_dict = dict()
    
    for key in embeddings_dict.keys():
            print(embeddings_dict[key], type(embeddings_dict[key]))
            if None not in embeddings_dict[key]:
                processed_dict[key] = embeddings_dict[key]
            # else:
                # embeddings_dict[key] = normalize(embeddings_dict[key].reshape(1, -1), 'l2')[0]
    #image_ids = embeddings_dict.keys()
    #embeddings_matrix = np.stack([embeddings_dict[img_id] for img_id in image_ids])

    return processed_dict

def save_cluster_distribution(clusters, ids):

    df_clusters = pd.DataFrame({
        'Image': ids,
        'Cluster': clusters,
    })

    np.save('kmeans_centers_umd_arcface.npy', np.array(kmeans.cluster_centers_)) #salvando os centros e a distribuição do cluster
    joblib.dump(kmeans, 'kmeans_umd_arcface.pkl')

    df_clusters.to_csv(f'umd_cluster_assignments_arcface_train.csv', index=False)

def plot_cluster_distribution(clusters):
    unique_clusters, counts = np.unique(clusters, return_counts=True)

    plt.bar(unique_clusters, counts, color='skyblue')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Embeddings')
    plt.title('Cluster Distribution')
    plt.xticks(unique_clusters)  # Ensure all clusters are labeled on the x-axis
    plt.show()

if __name__ == "__main__":
    embeddings_dict = preprocess()
    print(len(embeddings_dict.keys()))

    # kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    # clusters = kmeans.fit_predict(embeddings_matrix)

    # embeddings_dict = build_embedding_dict()
    # print(embeddings_dict.items())
    # np.savez_compressed("embeddings_umd_arcface_dict.npz", **embeddings_dict)
    # save_cluster_distribution(clusters, image_ids)
    # plot_cluster_distribution(clusters)

    