import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_distances

caminho_base = r"C:\Users\emily\Downloads\umdfaces_features\csv\train"
caminho_centroides = r"C:\Users\emily\Downloads\kmeans_centers_griaule_train.npy"
caminho_csv = r"C:\Users\emily\OneDrive\Documents\IC\parte 3 - testes\griaule\umd\DifferentPerson_train_umd.csv" 
# aqui eu usei pra different person, para same person é só mudar Folder A/B para Folder

#  centroides
centroides = np.load(caminho_centroides)  # shape (15, embedding_dim)

df = pd.read_csv(caminho_csv)

# carregar um embedding dado o folder e o nome do arquivo
def carregar_embedding(folder, image_name):
    caminho = os.path.join(caminho_base, str(folder), image_name)
    try:
        return pd.read_csv(caminho, header=None).values.flatten()
    except Exception as e:
        print(f"Erro ao carregar {caminho}: {e}")
        return None

# encontrar o centróide mais próximo
def encontrar_cluster_mais_proximo(embedding, centroides):
    if embedding is None:
        return "N/A"
    distancias = cosine_distances([embedding], centroides)[0]
    return int(np.argmin(distancias))

cluster_a = []
cluster_b = []

for idx, row in df.iterrows():
    emb_a = carregar_embedding(row["Folder A"], row["Image A"])
    emb_b = carregar_embedding(row["Folder B"], row["Image B"])

    cluster_a.append(encontrar_cluster_mais_proximo(emb_a, centroides))
    cluster_b.append(encontrar_cluster_mais_proximo(emb_b, centroides))

df["Cluster A"] = cluster_a
df["Cluster B"] = cluster_b

# Salvar o novo CSV
df.to_csv("csv_com_clusters.csv", index=False)
