import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    kmeans_centers = np.load("kmeans_centers_griaule_train.npy")
    print(kmeans_centers.shape)
    df_umd = pd.read_csv("umdfaces_embeddings_griaule_train.csv")
    umd_embeddings = np.array(df_umd.loc[:, "embedding_0":])
    umd_embeddings = normalize(umd_embeddings, "l2")
    print(umd_embeddings.shape)
    print(kmeans_centers.shape)
    distances = cdist(umd_embeddings, kmeans_centers, metric='cosine')
    closest_centers = np.argmin(distances, axis=1)
    print(closest_centers)
    df_result = pd.DataFrame({
        'id': df_umd.iloc[:, 1], 
        'closest_cluster': closest_centers
    })
    print(df_result.head)

    cluster_counts = df_result['closest_cluster'].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="crest")

    plt.title("Distribuição de Embeddings por Cluster")
    plt.xlabel("Índice do Cluster")
    plt.ylabel("Número de Embeddings")
    plt.xticks(cluster_counts.index)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
    # df_result.to_csv("umdfaces_closest_clusters_normalized.csv", index=False)
    
