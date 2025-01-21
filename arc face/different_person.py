import os
import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
from itertools import combinations
import random
from pathlib import Path

# Face analysis application using ArcFace iResNet100
app = FaceAnalysis(name='buffalo_l')  # 'buffalo_l' includes ArcFace iResNet100
app.prepare(ctx_id=0, det_thresh=0.5)  # ctx_id=0 for CPU, ctx_id=1 for GPU
parent_folder = r"C:\Users\emily\Downloads\RFW_dataset\separacao\clusters"

# Similarity threshold
similarity_threshold = 0.5 

# Get the embedding of an image
def get_embedding(image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)
    if len(faces) > 0:
        return faces[0].embedding  # Returns the embedding of the first detected face
    return None

# Compare two images
def compare_faces(img1_path, img2_path):
    emb1 = get_embedding(img1_path)
    emb2 = get_embedding(img2_path)
    
    if emb1 is None or emb2 is None:
        return None
    
    # Compute the cosine similarity between embeddings
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity

# Process images and generate the CSV
def process_images(base_directory, total_comparisons, csv_filename):
    num_clusters = 15
    comparisons_per_cluster = total_comparisons // num_clusters
    results = []
    comparison_count = 0

    for cluster_id in range(num_clusters):
        cluster_directory = os.path.join(base_directory, f'cluster_{cluster_id}')
        image_paths = list(Path(cluster_directory).glob('*.jpg'))

        # If there are too many images, sample a smaller subset
        if len(image_paths) > 1000:
            image_paths = random.sample(image_paths, 1000)

        # Randomly select pairs of images within the same cluster
        if len(image_paths) > 1:
            sampled_images = random.sample(image_paths, min(len(image_paths), 1000))
            sampled_pairs = random.sample(
                list(combinations(sampled_images, 2)), 
                min(comparisons_per_cluster, len(sampled_images) * (len(sampled_images) - 1) // 2)
            )

            for img1, img2 in sampled_pairs:
                similarity = compare_faces(img1, img2)
                if similarity is not None:
                    same_person = similarity > similarity_threshold
                    results.append([img1.name, img2.name, similarity, same_person, cluster_id])
                    
                # Stop if the maximum number of comparisons is reached
                if len(results) >= total_comparisons:
                    break
        if len(results) >= total_comparisons:
            break

    # Message indicating the total number of comparisons performed
    if len(results) < total_comparisons:
        print(f"Performed {len(results)} comparisons, which is less than the expected total of {total_comparisons}.")

    # Save to CSV
    df = pd.DataFrame(results, columns=['image A', 'image B', 'similarity', 'same person', 'cluster'])
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")

# Example usage
if __name__ == "__main__":
    base_directory = 'C:/Users/emily/Downloads/RFW_dataset/separacao/clusters'
    csv_filename = r"C:\Users\emily\OneDrive\Documents\IC\results\arcface\clusteringFalse.csv"
    total_comparisons = 6000  # Total number of comparisons

    # Save results
    process_images(base_directory, total_comparisons, csv_filename)
