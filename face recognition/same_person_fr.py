import os
import numpy as np
import random
import pandas as pd
from itertools import combinations
from pathlib import Path
import face_recognition

# Define the path to the cluster folder
parent_folder = r"C:\Users\emily\Downloads\RFW_dataset\separacao\clusters"

# Define the similarity threshold (adjust as needed)
similarity_threshold = 0.6  # Threshold to consider faces as the same person (lower value = higher similarity)
message_threshold = 100  # Number of comparisons after which a message is displayed

# Function to get the encoding of an image
def get_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        return encodings[0]  # Return the encoding of the first face found
    return None

# Function to compare two images
def compare_faces(img1_path, img2_path):
    enc1 = get_encoding(img1_path)
    enc2 = get_encoding(img2_path)
    
    if enc1 is None or enc2 is None:
        return None
    
    # Calculate the Euclidean distance between the encodings
    distance = np.linalg.norm(enc1 - enc2)
    return distance

# Function to process images and generate the CSV
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

            # Group images by the first 7 digits of their filename
            for image_path in image_paths:
                prefix = image_path.stem[:7]  # Consider only the first 7 digits
                if prefix not in groups:
                    groups[prefix] = []
                groups[prefix].append(image_path)

            # Compare only images with the same 7-digit prefix
            for prefix, images in groups.items():
                if len(images) < 2:
                    continue  # Skip if there are less than two images in the group

                # Generate all combinations of pairs of images with the same prefix
                for img1, img2 in combinations(images, 2):
                    distance = compare_faces(img1, img2)
                    if distance is not None:
                        same_person = distance < similarity_threshold
                        results.append([img1.name, img2.name, distance, same_person, cluster_id])

                    # Update the counter and check if it's time to display a message
                    comparison_count += 1
                    if comparison_count % message_threshold == 0:
                        print(f"{comparison_count} comparisons completed so far.")

                    # Check if the maximum number of comparisons per cluster is reached
                    if len(results) >= comparisons_per_cluster * (cluster_id + 1):
                        break

            if len(results) >= comparisons_per_cluster * num_clusters:
                break

    except KeyboardInterrupt:
        print("Process manually interrupted. Saving accumulated data...")

    # If the total number of comparisons is less than expected, adjust
    if len(results) < total_comparisons:
        print(f"Performed {len(results)} comparisons, which is less than the expected total of {total_comparisons}.")

    # Save results to CSV
    output_directory = os.path.dirname(csv_filename)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    df = pd.DataFrame(results, columns=['image A', 'image B', 'distance', 'same person', 'cluster'])
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to '{csv_filename}'")

# Example usage
if __name__ == "__main__":
    # Define cluster path, CSV name, and total number of comparisons
    base_directory = 'C:/Users/emily/Downloads/RFW_dataset/separacao/clusters'
    csv_filename = r'C:\Users\emily\OneDrive\Documents\IC\results\face_recognition\clusteringTrue.csv'
    total_comparisons = 600  # Total number of comparisons

    # Process images and save results
    process_images(base_directory, total_comparisons, csv_filename)
