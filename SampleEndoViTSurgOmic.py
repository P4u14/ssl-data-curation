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

def load_training_filenames():
    # Define the file path
    file_path = '/mnt/ceph/tco/TCO-All/SharedDatasets/SurgOmicFeatures/FeatureDetection/annotations.csv'

    # Initialize an empty list to store the filenames
    filenames = []

    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for index, line in enumerate(lines):

            # Skip the first line (index 0)
            if index == 0:
                continue

            parts = line.split(',')
            filename = parts[0]

            # Remove the .png suffix
            if filename.endswith('.png'):
                filename = filename[:-4]

            filenames.append(filename)

    return filenames

def load_embeddings(directory):
    training_filenames = load_training_filenames()
    embeddings = []
    paths = []
    loaded_images_counter = 0
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".pt") and filename[:-3] in training_filenames:
            # print(filename)
            loaded_images_counter = loaded_images_counter + 1
            print("Loaded image: " + str(loaded_images_counter))
            file_path = os.path.join(directory, filename)
            loaded_data = torch.load(file_path)
            print("Tensor: " )
            print(loaded_data)
            # Ensure the loaded tensor is 3D
            if loaded_data.ndim == 3:
                loaded_data = loaded_data.reshape(loaded_data.shape[0], -1)  # Flatten the middle dimension
            embeddings.append(loaded_data)
            paths.append(file_path)
        """ else:
            print("Aussortiert: " + filename) """
    # Convert all embeddings to torch tensors if they are not already
    embeddings = [torch.tensor(e) if not isinstance(e, torch.Tensor) else e for e in embeddings]
    return torch.cat(embeddings), paths



data_directory = '/mnt/cluster/workspaces/students/schreibpa/embeddings-small/endoViT/SurgOmicFeatures/'


#disable backpropagation (and other stuff)
with torch.inference_mode(): 
    # Load data
    data, paths_list = load_embeddings(data_directory)

    # Total number of images in the loaded dataset
    total_number_images = len(paths_list) # should be 14000 for SurgOmicFeatures

    # Print shape to verify it's 2D
    print("Data shape after flattening:", data.shape)

    # Ensure data is 2D
    if data.ndim != 2:
        raise ValueError(f"Data should be 2D but got shape {data.shape}")
    
    # Convert data to torch tensor and move to CUDA
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

    # Define the path to the "endoViT" folder
    endoViT_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'endoViT')

    # Create the directory if it does not exist
    os.makedirs(endoViT_dir, exist_ok=True)

    # Define the file path within the "endoViT" folder
    file_path = os.path.join(endoViT_dir, '10_percent_selected_image_paths.txt')

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

    # Define the file path within the "endoViT" folder
    file_path = os.path.join(endoViT_dir, '20_percent_selected_image_paths.txt')

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

    # Define the file path within the "endoViT" folder
    file_path = os.path.join(endoViT_dir, '30_percent_selected_image_paths.txt')

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

    # Define the file path within the "endoViT" folder
    file_path = os.path.join(endoViT_dir, '40_percent_selected_image_paths.txt')

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

    # Define the file path within the "endoViT" folder
    file_path = os.path.join(endoViT_dir, '50_percent_selected_image_paths.txt')

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

    # Define the file path within the "endoViT" folder
    file_path = os.path.join(endoViT_dir, '60_percent_selected_image_paths.txt')

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

    # Define the file path within the "endoViT" folder
    file_path = os.path.join(endoViT_dir, '70_percent_selected_image_paths.txt')

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

    # Define the file path within the "endoViT" folder
    file_path = os.path.join(endoViT_dir, '80_percent_selected_image_paths.txt')

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

    # Define the file path within the "endoViT" folder
    file_path = os.path.join(endoViT_dir, '90_percent_selected_image_paths.txt')

    # Open the file in write mode (or create the file if it does not exist)
    with open(file_path, 'w') as file:
        for path in selected_image_paths:
            file.write(path + '\n')

    print("End selection of 90\%")
