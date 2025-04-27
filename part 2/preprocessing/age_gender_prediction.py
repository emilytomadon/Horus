# AGE AND GENDER PREDICTION #

import os
import csv
from deepface import DeepFace

train_dir = r'C:\Users\emily\Downloads\RFWsummarized_split\RFWsummarized_split\train'
output_csv = r'C:\Users\emily\Downloads\RFWsummarized_split\age_gender_prediction.csv'

# list all images in a directory
def list_images_in_directory(directory):
    return [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# calculate age and gender using DeepFace
def get_age_gender(image_path):
    try:
        analysis = DeepFace.analyze(image_path, actions=['age', 'gender'], enforce_detection=False)
        age = analysis[0]['age']
        gender = analysis[0]['gender']
        return age, gender
    except Exception as e:
        print(f"Error analyzing the image {image_path}: {e}")
        return None, None

image_paths = [os.path.join(root, f)
               for root, _, files in os.walk(train_dir)
               for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

predictions = []

for image_path in image_paths:
    print(f"Analyzing the image {image_path}...")
    age, gender = get_age_gender(image_path)
    if age is not None and gender is not None:
        predictions.append([os.path.relpath(image_path, train_dir), age, gender])

with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image', 'Age', 'Gender'])  # Header
    writer.writerows(predictions)

print(f"Age and gender predictions saved in {output_csv}.")
