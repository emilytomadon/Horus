from sklearn.cluster import KMeans
import numpy as np

# Ensure embeddings are already extracted
# all_embeddings = np.array(all_embeddings)

n_clusters = 15

# clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(all_embeddings)

print("Clusters assigned to each embedding:", clusters)
