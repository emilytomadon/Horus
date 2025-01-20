import os
import numpy as np
import random
import pandas as pd
from itertools import combinations
from pathlib import Path
import face_recognition

parent_folder = r"C:\Users\emily\Downloads\RFW_dataset\separacao\clusters"

# Similarity threshold
similarity_threshold = 0.6  # Lower value = higher similarity in this case

# Get the encoding
def get_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        return encodings[0]  # Returns the encoding of the first face found
    return None

# Compare two images
def compare_faces(img1_path, img2_path):
    enc1 = get_encoding(img1_path)
    enc2 = get_encoding(img2_path)
    
    if enc1 is None or enc2 is None:
        return None
    
    # Calculate Euclidean distance between encodings
    distance = np.linalg.norm(enc1 - enc2)
    return distance

# Process images and generate CSV
def process_images(base_directory, total_comparisons, csv_filename):
    num_clusters = 15
    comparisons_per_cluster = total_comparisons // num_clusters
    results = []
    comparison_count = 0

    try:
        for cluster_id in range(num_clusters):
            cluster_directory = os.path.join(base_directory, f'cluster_{cluster_id}')
            image_paths = list(Path(cluster_directory).glob('*.jpg'))
            groups = {}

            # Group images by 7-digit prefix
            # Images with the same first 7 digits belong to the same person
            for image_path in image_paths:
                prefix = image_path.stem[:7]
                if prefix not in groups:
                    groups[prefix] = []
                groups[prefix].append(image_path)

            all_image_paths = [path for paths in groups.values() for path in paths]

            # If there are too many images, randomly sample a smaller subset
            if len(all_image_paths) > 1000:
                all_image_paths = random.sample(all_image_paths, 1000)

            # Compare images only from different 7-digit groups
            while len(results) < comparisons_per_cluster * (cluster_id + 1):
                if len(all_image_paths) < 2:
                    break

                # Select two random images from different groups
                img1_group, img2_group = random.sample(list(groups.keys()), 2)
                img1 = random.choice(groups[img1_group])
                img2 = random.choice(groups[img2_group])

                distance = compare_faces(img1, img2)
                if distance is not None:
                    same_person = distance < similarity_threshold
                    results.append([img1.name, img2.name, distance, same_person, cluster_id])

                # Check if the maximum number of comparisons per cluster has been reached
                if len(results) >= comparisons_per_cluster * num_clusters:
                    break

            if len(results) >= comparisons_per_cluster * num_clusters:
                break

    except KeyboardInterrupt:
        print("Process interrupted manually. Saving accumulated data...")

    # If the total number of comparisons is less than expected
    if len(results) < total_comparisons:
        print(f"Performed {len(results)} comparisons, which is less than the expected total of {total_comparisons}.")

    # save results to CSV
    output_directory = os.path.dirname(csv_filename)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    df = pd.DataFrame(results, columns=['Image A', 'Image B', 'Distance', 'Same Person', 'Cluster'])
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to '{csv_filename}'")

if __name__ == "__main__":
    base_directory = r'C:\Users\emily\Downloads\RFW_dataset\separacao\clusters'
    csv_filename = r'C:\Users\emily\OneDrive\Documents\IC\results\face_recognition\clusteringFalse_fr.csv'
    total_comparisons = 6000  # Total number of comparisons

    # Save results
    process_images(base_directory, total_comparisons, csv_filename)

