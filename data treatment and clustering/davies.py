from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

EMBEDDINGS_ARRAY_PATH = r'all_embeddings_griaule_train.npy'
CLUSTERS_MAX = 30

def kmeans_clusters(embeddings, n: int):
    kmeans = KMeans(n_clusters=n, random_state=10)
    clusters = kmeans.fit_predict(embeddings)
    return clusters

if __name__ == '__main__':
    embeddings_array = np.load(EMBEDDINGS_ARRAY_PATH)
    davies_bouldin_score_averages = []
    for i in range(4, CLUSTERS_MAX):
        cluster_labels = kmeans_clusters(embeddings_array, i)
        dscore_avg = davies_bouldin_score(embeddings_array, cluster_labels)
        davies_bouldin_score_averages.append(dscore_avg)
        print("For n_clusters =",i,"The average davies_bouldin_score is :", dscore_avg)

    davies_bouldin_score_averages_array = np.array(davies_bouldin_score_averages)
    np.save("davies-averages-rfw-griaule.npy", davies_bouldin_score_averages_array)

    fig=plt.subplots(figsize=(10,5))
    plt.plot(range(4, CLUSTERS_MAX), davies_bouldin_score_averages, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Davies-Bouldin Score')
    plt.xticks(range(4, CLUSTERS_MAX))
    plt.grid(True)
    plt.show()


    


