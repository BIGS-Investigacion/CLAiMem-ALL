import os
import shutil
import random
import math

def create_subdirectories_with_images(source_dir, dest_dir, n, seed):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    total_images = len(images)

    random.seed(seed)
    random.shuffle(images)

    images_per_folder = math.ceil(total_images/n)

    for i in range(n):
        sub_dir = os.path.join(dest_dir, f'subfolder_{i+1}')
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        start_index = i * images_per_folder
        end_index = start_index + images_per_folder if i < n - 1 else total_images

        for j in range(start_index, end_index):
            shutil.copy(os.path.join(source_dir, images[j]), sub_dir)

if __name__ == "__main__":
    source_directory = 'data/original/tcga_sample'
    destination_directory = 'data/original/tcga_sample_folders'
    num_folders = 5  # Change this to the desired number of subdirectories
    seed = 42  # Change this to the desired seed value

    create_subdirectories_with_images(source_directory, destination_directory, num_folders, seed)