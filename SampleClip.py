import sys
sys.path.insert(0, "..")

import torch
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import os

from src.clusters import HierarchicalCluster
from src import (
  hierarchical_kmeans_gpu as hkmg,
  hierarchical_sampling as hs
)
from sklearn.preprocessing import normalize

def load_embeddings(data_directory):
    embeddings = []
    paths = []
    for sub_directory in sub_directory_list:
        print('subdirectory ' + str(sub_directory))
        directory = data_directory + sub_directory
        for filename in sorted(os.listdir(directory), key=lambda x:int(x.split('.')[0])):
            if filename.endswith(".pt"):
                file_path = os.path.join(directory, filename)
                loaded_data = torch.load(file_path)
                print("Tensor: " )
                print(loaded_data)
                # Ensure the loaded tensor is 3D
                if loaded_data.ndim == 3:
                    loaded_data = loaded_data.reshape(loaded_data.shape[0], -1)  # Flatten the middle dimension
                embeddings.append(loaded_data)
                paths.append(file_path)
    # Convert all embeddings to torch tensors if they are not already
    embeddings = [torch.tensor(e) if not isinstance(e, torch.Tensor) else e for e in embeddings]
    return torch.cat(embeddings), paths


# The folder structure inside Cholec80
# This list contains all Cholec80 directories
""" sub_directory_list = [
                  '1/02', '1/04', '1/06', '1/12', '1/24', '1/29', '1/34', '1/37', '1/38', '1/39', 
                  '1/44', '1/58', '1/60', '1/61', '1/64', '1/66', '1/75', '1/78', '1/79', '1/80', 
                  '2/01', '2/03', '2/05', '2/09', '2/13', '2/16', '2/18', '2/21', '2/22', '2/25',
                  '2/31', '2/36', '2/45', '2/46', '2/48', '2/50', '2/62', '2/71', '2/72', '2/73',
                  '3/10', '3/15', '3/17', '3/20', '3/32', '3/41', '3/42', '3/43', '3/47', '3/49',
                  '3/51', '3/52', '3/53', '3/55', '3/56', '3/69', '3/70', '3/74', '3/76', '3/77',
                  '4/07', '4/08', '4/11', '4/14', '4/19', '4/23', '4/26', '4/27', '4/28', '4/30',
                  '4/33', '4/35', '4/40', '4/54', '4/57', '4/59', '4/63', '4/65', '4/67', '4/68'] """

# This list does not contain the test directories: 3, 24, 40, 16, 32, 34, 7, 14, 29, 8, 31, 37, 15, 30, 18, 11, 6, 2, 33, 21, 23, 25, 19, 20
sub_directory_list = [
                  '1/04', '1/12', '1/38', '1/39', 
                  '1/44', '1/58', '1/60', '1/61', '1/64', '1/66', '1/75', '1/78', '1/79', '1/80', 
                  '2/01', '2/05', '2/09', '2/13', '2/22',
                  '2/36', '2/45', '2/46', '2/48', '2/50', '2/62', '2/71', '2/72', '2/73',
                  '3/10', '3/17', '3/41', '3/42', '3/43', '3/47', '3/49',
                  '3/51', '3/52', '3/53', '3/55', '3/56', '3/69', '3/70', '3/74', '3/76', '3/77',
                  '4/26', '4/27', '4/28',
                  '4/35', '4/54', '4/57', '4/59', '4/63', '4/65', '4/67', '4/68']

data_directory = '/mnt/cluster/workspaces/students/schreibpa/embeddings-small/clip/Cholec80/'


#disable backpropagation (and other stuff)
with torch.inference_mode(): 
    # Load data
    data, paths_list = load_embeddings(data_directory)

    # Total number of images in the loaded dataset
    total_number_images = len(paths_list) # should be 184817 for Cholec80

    # Print shape to verify it's 2D
    print("Data shape after flattening:", data.shape)

    # Ensure data is 2D
    if data.ndim != 2:
        raise ValueError(f"Data should be 2D but got shape {data.shape}")
    
    # Normalize the features to unit length (cosine)
    #normalized_data = normalize(data, norm='l2')
    
    # Convert data to torch tensor and move to CUDA
    #data_tensor = torch.tensor(normalized_data, device="cuda", dtype=torch.float32)
    data_tensor = torch.tensor(data, device="cuda", dtype=torch.float32)

    print("Loaded data on GPU")

    clusters = hkmg.hierarchical_kmeans_with_resampling(
        data=data_tensor,
        n_clusters=[int(total_number_images / 200), int(total_number_images / 500), int(total_number_images / 1000)], #TODO: adjust
        n_levels=3, #TODO: adjust
        sample_sizes=[30, 15, 2], #TODO: adjust
        verbose=False,
    )

    # Free unused memory
    torch.cuda.empty_cache()

    print("Calculated Clusters")
    print(clusters)



    print("Start selection of 10\%")
    # Factor for percentage of how many images should be selected for the output
    selection_factor = 0.1

    cl = HierarchicalCluster.from_dict(clusters)
    print("Anzahl Bilder insgesamt: " + str(total_number_images))
    target_amount = int(total_number_images * selection_factor)
    print("Anzahl auszuwählender Bilder: " + str(target_amount))
    sampled_indices = hs.hierarchical_sampling(cl, target_size=target_amount)
    sampled_indices_as_int = sampled_indices.astype(int)
    sampled_points = data[sampled_indices_as_int]
    print("Sampled points")
    print(sampled_points.shape)
    print(sampled_points[0])
    print("Sampled indices")
    print(sampled_indices)

    print("Vergleich")
    print(sampled_indices[0])
    test_data = torch.load(paths_list[sampled_indices[0]])
    if test_data.ndim == 3:
                    loaded_data = test_data.reshape(test_data.shape[0], -1) 
    test_data = torch.tensor(test_data)
    # Reshape to torch.Size([151296])
    test_data = test_data.view(-1)
    print(test_data)


    # Get all paths of the selected images
    selected_image_paths = []
    for index in sampled_indices:
        selected_image_paths.append(paths_list[index])

    # Define the path to the "clip" folder
    clip_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clip')

    # Create the directory if it does not exist
    os.makedirs(clip_dir, exist_ok=True)

    # Define the file path within the "clip" folder
    file_path = os.path.join(clip_dir, '10_percent_selected_image_paths.txt')

    # Open the file in write mode (or create the file if it does not exist)
    with open(file_path, 'w') as file:
        for path in selected_image_paths:
            file.write(path + '\n')

    print("End selection of 10\%")



    print("Start selection of 20\%")
    # Factor for percentage of how many images should be selected for the output
    selection_factor = 0.2

    cl = HierarchicalCluster.from_dict(clusters)
    target_amount = int(total_number_images * selection_factor)
    sampled_indices = hs.hierarchical_sampling(cl, target_size=target_amount)

    # Get all paths of the selected images
    selected_image_paths = []
    for index in sampled_indices:
        selected_image_paths.append(paths_list[index])

    # Define the file path within the "clip" folder
    file_path = os.path.join(clip_dir, '20_percent_selected_image_paths.txt')

    # Open the file in write mode (or create the file if it does not exist)
    with open(file_path, 'w') as file:
        for path in selected_image_paths:
            file.write(path + '\n')

    print("End selection of 20\%")



    print("Start selection of 30\%")
    # Factor for percentage of how many images should be selected for the output
    selection_factor = 0.3

    cl = HierarchicalCluster.from_dict(clusters)
    target_amount = int(total_number_images * selection_factor)
    sampled_indices = hs.hierarchical_sampling(cl, target_size=target_amount)

    # Get all paths of the selected images
    selected_image_paths = []
    for index in sampled_indices:
        selected_image_paths.append(paths_list[index])

    # Define the file path within the "clip" folder
    file_path = os.path.join(clip_dir, '30_percent_selected_image_paths.txt')

    # Open the file in write mode (or create the file if it does not exist)
    with open(file_path, 'w') as file:
        for path in selected_image_paths:
            file.write(path + '\n')

    print("End selection of 30\%")



    print("Start selection of 40\%")
    # Factor for percentage of how many images should be selected for the output
    selection_factor = 0.4

    cl = HierarchicalCluster.from_dict(clusters)
    target_amount = int(total_number_images * selection_factor)
    sampled_indices = hs.hierarchical_sampling(cl, target_size=target_amount)

    # Get all paths of the selected images
    selected_image_paths = []
    for index in sampled_indices:
        selected_image_paths.append(paths_list[index])

    # Define the file path within the "clip" folder
    file_path = os.path.join(clip_dir, '40_percent_selected_image_paths.txt')

    # Open the file in write mode (or create the file if it does not exist)
    with open(file_path, 'w') as file:
        for path in selected_image_paths:
            file.write(path + '\n')

    print("End selection of 40\%")



    print("Start selection of 50\%")
    # Factor for percentage of how many images should be selected for the output
    selection_factor = 0.5

    cl = HierarchicalCluster.from_dict(clusters)
    target_amount = int(total_number_images * selection_factor)
    sampled_indices = hs.hierarchical_sampling(cl, target_size=target_amount)

    # Get all paths of the selected images
    selected_image_paths = []
    for index in sampled_indices:
        selected_image_paths.append(paths_list[index])

    # Define the file path within the "clip" folder
    file_path = os.path.join(clip_dir, '50_percent_selected_image_paths.txt')

    # Open the file in write mode (or create the file if it does not exist)
    with open(file_path, 'w') as file:
        for path in selected_image_paths:
            file.write(path + '\n')

    print("End selection of 50\%")



    print("Start selection of 60\%")
    # Factor for percentage of how many images should be selected for the output
    selection_factor = 0.6

    cl = HierarchicalCluster.from_dict(clusters)
    target_amount = int(total_number_images * selection_factor)
    sampled_indices = hs.hierarchical_sampling(cl, target_size=target_amount)

    # Get all paths of the selected images
    selected_image_paths = []
    for index in sampled_indices:
        selected_image_paths.append(paths_list[index])

    # Define the file path within the "clip" folder
    file_path = os.path.join(clip_dir, '60_percent_selected_image_paths.txt')

    # Open the file in write mode (or create the file if it does not exist)
    with open(file_path, 'w') as file:
        for path in selected_image_paths:
            file.write(path + '\n')

    print("End selection of 60\%")



    print("Start selection of 70\%")
    # Factor for percentage of how many images should be selected for the output
    selection_factor = 0.7

    cl = HierarchicalCluster.from_dict(clusters)
    target_amount = int(total_number_images * selection_factor)
    sampled_indices = hs.hierarchical_sampling(cl, target_size=target_amount)

    # Get all paths of the selected images
    selected_image_paths = []
    for index in sampled_indices:
        selected_image_paths.append(paths_list[index])

    # Define the file path within the "clip" folder
    file_path = os.path.join(clip_dir, '70_percent_selected_image_paths.txt')

    # Open the file in write mode (or create the file if it does not exist)
    with open(file_path, 'w') as file:
        for path in selected_image_paths:
            file.write(path + '\n')

    print("End selection of 70\%")



    print("Start selection of 80\%")
    # Factor for percentage of how many images should be selected for the output
    selection_factor = 0.8

    cl = HierarchicalCluster.from_dict(clusters)
    target_amount = int(total_number_images * selection_factor)
    sampled_indices = hs.hierarchical_sampling(cl, target_size=target_amount)

    # Get all paths of the selected images
    selected_image_paths = []
    for index in sampled_indices:
        selected_image_paths.append(paths_list[index])

    # Define the file path within the "clip" folder
    file_path = os.path.join(clip_dir, '80_percent_selected_image_paths.txt')

    # Open the file in write mode (or create the file if it does not exist)
    with open(file_path, 'w') as file:
        for path in selected_image_paths:
            file.write(path + '\n')

    print("End selection of 80\%")



    print("Start selection of 90\%")
    # Factor for percentage of how many images should be selected for the output
    selection_factor = 0.9

    cl = HierarchicalCluster.from_dict(clusters)
    target_amount = int(total_number_images * selection_factor)
    sampled_indices = hs.hierarchical_sampling(cl, target_size=target_amount)

    # Get all paths of the selected images
    selected_image_paths = []
    for index in sampled_indices:
        selected_image_paths.append(paths_list[index])

    # Define the file path within the "clip" folder
    file_path = os.path.join(clip_dir, '90_percent_selected_image_paths.txt')

    # Open the file in write mode (or create the file if it does not exist)
    with open(file_path, 'w') as file:
        for path in selected_image_paths:
            file.write(path + '\n')

    print("End selection of 90\%")
