import os
import random
import csv
import pandas as pd
from itertools import combinations
import numpy as np

EMBEDDINGS_PATH = r'/home/gimicaroni/Documents/Datasets/rfw_embeddings_griaule/features/train/'

def get_embedding(image_path):
    emb = pd.read_csv(EMBEDDINGS_PATH + image_path, header=None).values.flatten()
    return np.array(emb)

def cosine_distance(img1, img2):
    emb1 = get_embedding(img1)
    emb2 = get_embedding(img2)
    if emb1 is None or emb2 is None:
        return None

    # Compute the cosine similarity between embeddings
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return 1-similarity

def generate_random_pairs(images, condition_func, num_pairs):
    possible_pairs = [pair for pair in combinations(images, 2) if condition_func(pair[0], pair[1])]
    random.shuffle(possible_pairs)
    return possible_pairs[:num_pairs]

# Condition to different person
def same_first_digit(img_a, img_b):
    return img_a[0] == img_b[0]

# Condition to same person
def same_seven_digits(img_a, img_b):
    return img_a[:7] == img_b[:7]

def generate_pairs_from_csv(df, total_pairs):
    all_pairs_dp, all_pairs_sp = [], []
    clusters = df['Cluster'].unique()
    pairs_per_cluster = total_pairs // len(clusters)
    
    for cluster in clusters:
        images = df[df['Cluster'] == cluster]['Image'].tolist()
        if len(images) >= 2:
            dp_pairs = generate_random_pairs(images, same_first_digit, pairs_per_cluster)
            sp_pairs = generate_random_pairs(images, same_seven_digits, pairs_per_cluster)
            for img_a, img_b in dp_pairs:
                all_pairs_dp.append((img_a, img_b, cluster, cosine_distance(img_a, img_b)))
            for img_a, img_b in sp_pairs:
                all_pairs_sp.append((img_a, img_b, cluster, cosine_distance(img_a, img_b)))
    return all_pairs_dp, all_pairs_sp

def save_pairs_to_csv(pairs, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image A', 'Image B', 'Cluster', 'Cosine Distance'])
        writer.writerows(pairs)

# Exemplo de uso:
input_csv = 'cluster_assignments_griaule_train_k15.csv'  # Caminho para seu CSV de entrada
df = pd.read_csv(input_csv)

dp_pairs, sp_pairs = generate_pairs_from_csv(df, total_pairs=5000)

save_pairs_to_csv(sp_pairs, 'SamePerson_train_fromcsv.csv')
save_pairs_to_csv(dp_pairs, 'DifferentPerson_train_fromcsv.csv')

print("CSV files generated successfully.")