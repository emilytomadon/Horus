from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
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
    calinski_harabasz_score_averages = []
    for i in range(4, CLUSTERS_MAX):
        cluster_labels = kmeans_clusters(embeddings_array, i)
        cscore_avg = calinski_harabasz_score(embeddings_array, cluster_labels)
        calinski_harabasz_score_averages.append(cscore_avg)
        print("For n_clusters =",i,"The average calinski_score is :", cscore_avg)

    calinski_harabasz_score_averages_array = np.array(calinski_harabasz_score_averages)
    np.save("calinski-averages-rfw-griaule.npy", calinski_harabasz_score_averages_array)

    fig=plt.subplots(figsize=(10,5))
    plt.plot(range(4, CLUSTERS_MAX), calinski_harabasz_score_averages, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Calinski-Harabasz Score')
    plt.xticks(range(4, CLUSTERS_MAX))
    plt.grid(True)
    plt.show()


    


