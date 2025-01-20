# RENAME FILE

# Renaming the files for clustering
# For the 30% division, I used up to folder indian 991, caucasian 816, asian 818, and african 984 for training.

import os
import shutil

source_folder_base = r'C:\Users\emily\Downloads\RFW_dataset\separacao\train\indian' # later replace with asian, african, and caucasian
destination_folder = r'C:\Users\emily\Downloads\RFW_dataset\separacao\total'

global_photo_counter = 1

# Loop from 00001 to 01000
for i in range(1, 1000):
    # Formats the subfolder number
    subfolder_name = f'{i:05d}'
    subfolder_path = os.path.join(source_folder_base, subfolder_name)

    # If it exists
    if os.path.exists(subfolder_path):
        for filename in os.listdir(subfolder_path):
            source_file = os.path.join(subfolder_path, filename)
            # New file name includes the subfolder number and the photo number
            new_filename = f'in{subfolder_name}_{global_photo_counter:04d}.jpg' # in = indian, as = asian, wh = caucasian, and bl = african
            destination_file = os.path.join(destination_folder, new_filename)
            shutil.copy2(source_file, destination_file)
            global_photo_counter += 1
   # example: photo 3 in subfolder 500 indian would be renamed to in00500_1715 (since it was the 1715th photo processed)

        print(f'Files copied and renamed')
    else:
        print(f'Subfolder {subfolder_name} not found')
