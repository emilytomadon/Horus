# CALCULATES DISTANCES FOR TRAINING (ARCFACE) #

import os
import csv
import numpy as np
from scipy.spatial.distance import cosine
import insightface
from PIL import Image

train_dir = r'C:\Users\emily\Downloads\RFWsummarized_split\RFWsummarized_split\train\clusters'

model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=-1)  # CPU

def get_embedding(image_path):
    img = np.array(Image.open(image_path).convert('RGB'))
    faces = model.get(img)
    return faces[0].embedding if faces else None

def calculate_distance(embedding_a, embedding_b):
    return cosine(embedding_a, embedding_b)

def update_csv_with_distances(input_csv_path, output_csv_path):
    with open(input_csv_path, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

    # Update pairs with cosine distances
    updated_rows = []
    header = rows[0]
    if len(header) == 2:
        updated_rows.append(['Image A', 'Image B', 'Cosine Distance'])
    else:
        updated_rows.append(['Image A', 'Image B', 'Cluster', 'Cosine Distance'])

    for row in rows[1:]:
        img_a, img_b = row[0], row[1]
        cluster = row[2] if len(row) > 2 else None

        # Build the full paths correctly
        if cluster is not None:
            img_a_path = os.path.join(train_dir, f'cluster_{cluster}', img_a)
            img_b_path = os.path.join(train_dir, f'cluster_{cluster}', img_b)
        else:
            img_a_path = os.path.join(train_dir, img_a)
            img_b_path = os.path.join(train_dir, img_b)
        
        embedding_a = get_embedding(img_a_path)
        embedding_b = get_embedding(img_b_path)

        if embedding_a is not None and embedding_b is not None:
            dist = calculate_distance(embedding_a, embedding_b)
            if cluster is not None:
                updated_rows.append([img_a, img_b, cluster, dist])
            else:
                updated_rows.append([img_a, img_b, dist])

    with open(output_csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(updated_rows)

    print(f"Cosine distances calculated and saved to {output_csv_path} successfully.")

sp_csv_path = r'C:\Users\emily\OneDrive\Documents\IC\algorithms\SamePerson_train.csv'
dp_csv_path = r'C:\Users\emily\OneDrive\Documents\IC\algorithms\DifferentPerson_train.csv'

sp_out_csv_path = r'C:\Users\emily\Downloads\RFWsummarized_split\RFWsummarized_split\train\clusters\SamePerson_train_arcface.csv'
dp_out_csv_path = r'C:\Users\emily\Downloads\RFWsummarized_split\RFWsummarized_split\train\clusters\DifferentPerson_train_arcface.csv'

update_csv_with_distances(sp_csv_path, sp_out_csv_path)
update_csv_with_distances(dp_csv_path, dp_out_csv_path)
