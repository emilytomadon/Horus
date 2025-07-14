import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns

KMEANS_CENTERS = r"kmeans_centers_griaule_train.npy"
EMBEDDINGS_CSV = r"umdfaces_embeddings_griaule_train2.csv" #csv com todos os ids na primeira coluna e os embeddings a partir da segunda coluna
ATTRIBUTION_THRESHOLD = 0.9

def plot_distribution(df_resultado):
    cluster_counts = df_resultado['closest_cluster'].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="crest")

    plt.title("Distribuição de Embeddings por Cluster")
    plt.xlabel("Índice do Cluster")
    plt.ylabel("Número de Embeddings")
    plt.xticks(cluster_counts.index)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def load_and_preproccess_embeddings(emb_csv_path):

    df_umd = pd.read_csv(emb_csv_path)
    umd_embeddings = np.array(df_umd.loc[:, "embedding_0":])
    umd_embeddings = normalize(umd_embeddings, "l2")

    return df_umd, umd_embeddings

def compute_distances_with_threshold(embeddings, kmeans_centers):
    distances = cdist(embeddings, kmeans_centers, metric='cosine')
    closest_clusters = np.argmin(distances, axis=1) #clusters associados as menores distâncias para cada embedding
    sorted_distances = np.sort(distances, axis=1)
    distance_ratios = sorted_distances[:, 0] / sorted_distances[:, 1]
    idx_assigned = np.where(distance_ratios < ATTRIBUTION_THRESHOLD)[0]
    return closest_clusters, idx_assigned


if __name__ == "__main__":

    kmeans_centers = np.load(KMEANS_CENTERS)
    df_umd, umd_embeddings = load_and_preproccess_embeddings(EMBEDDINGS_CSV)

    closest_clusters, idx_assigned = compute_distances_with_threshold(umd_embeddings, kmeans_centers)
    
    df_result = pd.DataFrame({
        'id': df_umd.iloc[idx_assigned, 1],
        'closest_cluster': closest_clusters[idx_assigned]
    })
    
    print(df_result.head())
    #plot_distribution(df_result)
    df_result.to_csv("umdfaces_closest_clusters_filtered.csv", index=False)
