import os
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import shutil

folder_path = r"C:\Users\emily\Downloads\RFW_dataset\separacao\total"

# all image paths
all_image_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Reduce to 2D with t-SNE
tsne = TSNE(n_components=2, random_state=0)
embeddings_2d = tsne.fit_transform(all_embeddings)

# apply K-means to the reduced data
n_clusters = 15
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
clusters = kmeans.fit_predict(embeddings_2d)

# Create folders for the clusters and move the images
output_dir = r"C:\Users\emily\Downloads\RFW_dataset\separacao\clusters"

for i in range(n_clusters):
    cluster_folder = os.path.join(output_dir, f'cluster_{i}')
    os.makedirs(cluster_folder, exist_ok=True)

for img_path, cluster_label in zip(all_image_paths, clusters):
    dest_path = os.path.join(output_dir, f'cluster_{cluster_label}', os.path.basename(img_path))
    shutil.copy(img_path, dest_path)

# plot clusters
plt.figure(figsize=(8, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap='tab20', alpha=0.7)
plt.xlabel('Vector 1')
plt.ylabel('Vector 2')
plt.title('T-SNE + K-means')
plt.colorbar()
plt.show()
