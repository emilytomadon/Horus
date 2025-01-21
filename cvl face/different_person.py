# Using CVL is more complicated and requires following a specific structure. So, for each cluster:
# I renamed the folder to "image" (which is necessary), run the code, and then use CVL. Afterward, I renamed the folder back to the cluster name.
# I renamed the CSV that was generated (e.g., the result of cluster 3 was result3) and repeat the process with the next cluster. Finally, I just merge the CSVs.

import os
import random
import pandas as pd
from pathlib import Path
from itertools import combinations

# Process images and generate CSV in the desired format
def generate_pairs_csv(cluster_directory, csv_filename, total_pairs=400):
    image_paths = list(Path(cluster_directory).glob('*.jpg'))

    # All possible pairs of images
    possible_pairs = list(combinations(image_paths, 2))

    # Filter pairs with the first 7 digits being the same
    filtered_pairs = [
        (img1, img2) for img1, img2 in possible_pairs 
        if img1.stem[:7] != img2.stem[:7]  # Images with different 7-digit prefixes
    ]

    # If the number of filtered pairs is less than the total desired pairs, adjust
    if len(filtered_pairs) < total_pairs:
        print(f"There are only {len(filtered_pairs)} pairs available after filtering. Using all of them.")
        selected_pairs = filtered_pairs
    else:
        # Randomly select the desired number of pairs
        selected_pairs = random.sample(filtered_pairs, total_pairs)

    # Create DataFrame
    data = []
    for idx, (img1, img2) in enumerate(selected_pairs):
        data.append([idx, img1.name, img2.name])

    df = pd.DataFrame(data, columns=['Index', 'A', 'B'])
    df.to_csv(csv_filename, index=False)
    print(f"CSV generated at {csv_filename}")

# Example
if __name__ == "__main__":
    # Define cluster path and CSV file name
    cluster_directory = r"C:\Users\emily\Downloads\RFW_dataset\separacao\clusters\images"  # Choose the desired cluster here
    csv_filename = r"C:\Users\emily\Downloads\RFW_dataset\separacao\clusters\pairs.csv"

    generate_pairs_csv(cluster_directory, csv_filename, total_pairs=400)
