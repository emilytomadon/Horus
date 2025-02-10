import os
import shutil
import random

# Define paths
dataset_path = "/home/gimicaroni/Documents/Datasets/RFWsummarized"  # Folder with all images
output_dir = "/home/gimicaroni/Documents/Datasets/RFWsummarized_split"  # Folder where split data will be saved

# Define split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Create output directories
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
test_dir = os.path.join(output_dir, "test")

for dir in [train_dir, val_dir, test_dir]:
    os.makedirs(dir, exist_ok=True)

# Get all image filenames
image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Ensure the split is reproducible
random.seed(123)  # Set fixed seed for consistency
random.shuffle(image_files)  # Shuffle once for a consistent split

# Compute split sizes
total_images = len(image_files)
train_end = int(total_images * train_ratio)
val_end = train_end + int(total_images * val_ratio)

# Split dataset
train_files = image_files[:train_end]
val_files = image_files[train_end:val_end]
test_files = image_files[val_end:]

# Function to move files to respective directories
def move_files(file_list, target_dir):
    for file in file_list:
        shutil.move(os.path.join(dataset_path, file), os.path.join(target_dir, file))

# Move images
move_files(train_files, train_dir)
move_files(val_files, val_dir)
move_files(test_files, test_dir)

print(f"Images successfully split into {output_dir}/train, {output_dir}/val, and {output_dir}/test")
