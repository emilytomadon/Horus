#GENERATE PAIRS #

import os
import random
import csv
from itertools import combinations

def list_images_in_directory(directory):
    return [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

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

def generate_pairs_for_train(directory, total_pairs):
    all_pairs_dp, all_pairs_sp = [], []
    clusters = [f for f in os.listdir(directory) if f.startswith('cluster_')]
    pairs_per_cluster = total_pairs // len(clusters)
    
    for cluster in clusters:
        cluster_dir = os.path.join(directory, cluster)
        images = list_images_in_directory(cluster_dir)
        
        if len(images) >= 2:
            dp_pairs = generate_random_pairs(images, same_first_digit, pairs_per_cluster)
            sp_pairs = generate_random_pairs(images, same_seven_digits, pairs_per_cluster)
            
            cluster_number = int(cluster.replace('cluster_', ''))
            
            for img_a, img_b in dp_pairs:
                all_pairs_dp.append((img_a, img_b, cluster_number))
            for img_a, img_b in sp_pairs:
                all_pairs_sp.append((img_a, img_b, cluster_number))
    
    return all_pairs_dp, all_pairs_sp

def generate_pairs_for_val(directory, num_pairs):
    images = list_images_in_directory(directory)
    dp_pairs = generate_random_pairs(images, same_first_digit, num_pairs)
    sp_pairs = generate_random_pairs(images, same_seven_digits, num_pairs)
    return dp_pairs, sp_pairs

def save_pairs_to_csv(pairs, filename, add_cluster=False):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        if add_cluster:
            writer.writerow(['Image A', 'Image B', 'Cluster'])
        else:
            writer.writerow(['Image A', 'Image B'])
        writer.writerows(pairs)

train_dir = r'C:\Users\emily\Downloads\RFWsummarized_split\RFWsummarized_split\train\clusters'
val_dir = r'C:\Users\emily\Downloads\RFWsummarized_split\RFWsummarized_split\val'

dp_train_pairs, sp_train_pairs = generate_pairs_for_train(train_dir, total_pairs=5000)
dp_val_pairs, sp_val_pairs = generate_pairs_for_val(val_dir, num_pairs=5000)

save_pairs_to_csv(sp_train_pairs, os.path.join(train_dir, 'SamePerson_train.csv'), add_cluster=True)
save_pairs_to_csv(dp_train_pairs, os.path.join(train_dir, 'DifferentPerson_train.csv'), add_cluster=True)
save_pairs_to_csv(sp_val_pairs, os.path.join(val_dir, 'SamePerson_val.csv'))
save_pairs_to_csv(dp_val_pairs, os.path.join(val_dir, 'DifferentPerson_val.csv'))

print("CSV files generated successfully.")
