import os
from deepface import DeepFace
import numpy as np

# load images
def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder, filename)
            images.append(img_path)
            filenames.append(filename)
    return images, filenames

# extract embeddings
def extract_embeddings(images):
    embeddings = []
    for idx, img in enumerate(images):
        try:
            representation = DeepFace.represent(img_path=img, model_name="Facenet", enforce_detection=False)
            embedding = representation[0]['embedding']
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error processing {img}: {e}")
            continue
    return embeddings

# Directory containing the images
folder_path = r"C:\Users\emily\Downloads\RFW_dataset\separacao\total"

# Load images from the folder
images, filenames = load_images_from_folder(folder_path)

# Extract embeddings
all_embeddings = extract_embeddings(images)

# Convert embeddings to a NumPy array
all_embeddings = np.array(all_embeddings)

print("Extraction complete. Total embeddings extracted:", len(all_embeddings))
