from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

EMBEDDINGS_ARRAY_PATH = '/home/gimicaroni/Documents/Unicamp/IC/HorusProjeto/HorusEthnicity/Horus/embeddings-array-rfw.npy'
CLUSTERS_MAX = 20

def kmeans_clusters(embeddings, n: int):
    kmeans = KMeans(n_clusters=n, random_state=10)
    clusters = kmeans.fit_predict(embeddings)
    return clusters

if __name__ == '__main__':
    embeddings_array = np.load(EMBEDDINGS_ARRAY_PATH)
    silhouette_averages = []
    for i in range(4, CLUSTERS_MAX):
        cluster_labels = kmeans_clusters(embeddings_array, i)
        silhouette_avg = silhouette_score(embeddings_array, cluster_labels)
        silhouette_averages.append(silhouette_avg)
        print("For n_clusters =",i,"The average silhouette_score is :", silhouette_avg)

    silhouette_averages_array = np.array(silhouette_averages)
    np.save("silhouette-averages-rfw.npy", silhouette_averages_array)

    fig=plt.subplots(figsize=(10,5))
    plt.plot(range(4, CLUSTERS_MAX), silhouette_averages, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette')
    plt.grid(True)
    plt.show()


    


