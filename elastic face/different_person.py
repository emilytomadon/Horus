# ELASTICFACE DIFFERENT PERSON

import os
from itertools import combinations
from pathlib import Path
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
import random

# Image preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

# Get the embedding of an image
def get_embedding(image_path, model):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        model.eval()
        embedding = model(image_tensor)
    return embedding

# Compare two images
def compare_images(image_path1, image_path2, model):
    embedding1 = get_embedding(image_path1, model)
    embedding2 = get_embedding(image_path2, model)
    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return similarity.item()

# Process images in a directory and generate a CSV
def process_images(base_directory, model, csv_filename, total_comparisons):
    num_clusters = 15
    comparisons_per_cluster = total_comparisons // num_clusters
    results = []
    comparison_count = 0
    message_threshold = 100  # Number of comparisons after which a message is displayed

    # Process each cluster folder
    for cluster_id in range(num_clusters):
        cluster_directory = os.path.join(base_directory, f'cluster_{cluster_id}')
        image_paths = list(Path(cluster_directory).glob('*.jpg'))

        # Group images by 7-digit prefix
        groups = {}
        for image_path in image_paths:
            prefix = image_path.stem[:7]
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append(image_path)

        all_image_paths = [path for paths in groups.values() for path in paths]

        # If there are too many images, sample a smaller amount
        if len(all_image_paths) > 1000:  # Adjust the number as needed
            all_image_paths = random.sample(all_image_paths, 1000)

        # Random pairs of images for comparison
        sampled_pairs = random.sample(list(combinations(all_image_paths, 2)), min(comparisons_per_cluster, len(all_image_paths)*(len(all_image_paths)-1)//2))

        # Compare images with different 7-digit prefixes
        for img1, img2 in sampled_pairs:
            if img1.stem[:7] != img2.stem[:7]:  # Compare only if the first 7 digits are different
                distance = compare_images(img1, img2, model)
                results.append([img1.name, img2.name, distance, 0, cluster_id])

                # Stop if the maximum number of comparisons is reached
                if len(results) >= total_comparisons:
                    break
        if len(results) >= total_comparisons:
            break

    # Message indicating the total number of comparisons performed
    if len(results) < total_comparisons:
        print(f"Performed {len(results)} comparisons, which is less than the expected total of {total_comparisons}.")

    # Save to CSV
    df = pd.DataFrame(results, columns=['image A', 'image B', 'distance', 'same person', 'cluster'])
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")

# Example
if __name__ == "__main__":
    model = iresnet100(pretrained=False)
    model.load_state_dict(torch.load('C:/Users/emily/Downloads/295672backbone.pth', map_location='cpu'))  # Model downloaded from ElasticFace GitHub

    base_directory = 'C:/Users/emily/Downloads/RFW_dataset/separacao/clusters'
    csv_filename = 'C:/Users/emily/OneDrive/Documents/IC/resultados/elasticface/clusterizacaoFalse_elasticface.csv'
    total_comparisons = 6000  # Total number of comparisons

    # Save results
    process_images(base_directory, model, csv_filename, total_comparisons)
