import os
import re
import matplotlib.pyplot as plt
import pandas as pd

def count_tools_in_annotation(annotation_file, image_filename):
    # Initialize counts for all tools
    tool_counts = {"AzygosVein": 0, "GastricTube": 0,
                   "VesselSealer": 0, "PermanentCauteryHook": 0, "ClipApplier": 0, 
                   "LargeClipApplier": 0, "Scissors": 0, "Suction": 0}
    
    # Extract image name without extension
    image_name = os.path.basename(image_filename).split('.')[0]
    
    # Open and read the annotation file
    with open(annotation_file, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:  # Skip the header
            parts = line.strip().split(',')
            if len(parts) == 11:
                file_name = parts[0].split('.')[0]
                if file_name == image_name:
                    # Update tool counts based on annotation
                    tool_counts["AzygosVein"] = int(parts[1] == 'True')
                    tool_counts["GastricTube"] = int(parts[2] == 'True') 
                    tool_counts["VesselSealer"] = int(parts[5] == 'True')
                    tool_counts["PermanentCauteryHook"] = int(parts[6] == 'True')
                    tool_counts["ClipApplier"] = int(parts[7] == 'True')
                    tool_counts["LargeClipApplier"] = int(parts[8] == 'True')
                    tool_counts["Scissors"] = int(parts[9] == 'True')
                    tool_counts["Suction"] = int(parts[10] == 'True')
                    break
    return tool_counts

def process_images_file(images_file_path, annotations_file):
    print(f'Start evaluating {images_file_path}')
    # Initialize totals for all tools
    tool_totals = {"AzygosVein": 0, "GastricTube": 0, 
                   "VesselSealer": 0, "PermanentCauteryHook": 0, "ClipApplier": 0, 
                   "LargeClipApplier": 0, "Scissors": 0, "Suction": 0}

    # Open and read the images file
    with open(images_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                # Extract the image file path
                match = re.search(r'/([^/]+)\.pt$', line)
                if match:
                    image_filename = match.group(1) + '.png'
                    # Determine the annotation file and count tools
                    if os.path.exists(annotations_file):
                        tool_counts = count_tools_in_annotation(annotations_file, image_filename)
                        for tool, count in tool_counts.items():
                            tool_totals[tool] += count

    return tool_totals

def plot_tool_counts(tool_counts_over_time, save_path):
    percentages = [10 * i for i in range(1, 11)]
    tools = ["AzygosVein", "GastricTube", "VesselSealer", 
             "PermanentCauteryHook", "ClipApplier", "LargeClipApplier", "Scissors", "Suction"]
    
    # Initialize the plot
    plt.figure(figsize=(12, 8))
    
    # Plot each tool's counts
    for tool in tools:
        counts = [tool_counts.get(tool, 0) for tool_counts in tool_counts_over_time]
        plt.plot(percentages, counts, marker='o', label=tool)
    
    # Add labels and title
    plt.xlabel('Percentage of Images Selected')
    plt.ylabel('Total Count of Each Tool')
    plt.title('Tool Counts by Percentage of Images Selected')
    plt.legend()
    plt.grid(True)
    plt.xticks(percentages)
    
    # Ensure the destination folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(save_path)
    print(f"Plot saved as {save_path}")
    plt.close()

def main():
    # Paths and files
    annotations_file = '/mnt/ceph/tco/TCO-All/SharedDatasets/SurgOmicFeatures/FeatureDetection/annotations.csv'
    image_files = [
        '10_percent_selected_image_paths.txt',
        '20_percent_selected_image_paths.txt',
        '30_percent_selected_image_paths.txt',
        '40_percent_selected_image_paths.txt',
        '50_percent_selected_image_paths.txt',
        '60_percent_selected_image_paths.txt',
        '70_percent_selected_image_paths.txt',
        '80_percent_selected_image_paths.txt',
        '90_percent_selected_image_paths.txt',
        '100_percent_selected_image_paths.txt'
    ]

    folder = 'clip_SurgOmicFeatures_no_test_images/'
    tool_counts_over_time = []

    for image_file in image_files:
        images_file_path = os.path.join(folder, image_file)
        tool_totals = process_images_file(images_file_path, annotations_file)
        tool_counts_over_time.append(tool_totals)

    save_path = 'plots/clip_SurgOmicFeatures_no_test_images.png'
    plot_tool_counts(tool_counts_over_time, save_path)




    folder = 'dinoV2_SurgOmicFeatures_no_test_images/'
    tool_counts_over_time = []

    for image_file in image_files:
        images_file_path = os.path.join(folder, image_file)
        tool_totals = process_images_file(images_file_path, annotations_file)
        tool_counts_over_time.append(tool_totals)

    save_path = 'plots/dinoV2_SurgOmicFeatures_no_test_images.png'
    plot_tool_counts(tool_counts_over_time, save_path)




    folder = 'endoViT_SurgOmicFeatures_no_test_images/'
    tool_counts_over_time = []

    for image_file in image_files:
        images_file_path = os.path.join(folder, image_file)
        tool_totals = process_images_file(images_file_path, annotations_file)
        tool_counts_over_time.append(tool_totals)

    save_path = 'plots/endoViT_SurgOmicFeatures_no_test_images.png'
    plot_tool_counts(tool_counts_over_time, save_path)




    folder = 'endoViTAlex_SurgOmicFeatures_no_test_images/'
    tool_counts_over_time = []

    for image_file in image_files:
        images_file_path = os.path.join(folder, image_file)
        tool_totals = process_images_file(images_file_path, annotations_file)
        tool_counts_over_time.append(tool_totals)

    save_path = 'plots/endoViTAlex_SurgOmicFeatures_no_test_images.png'
    plot_tool_counts(tool_counts_over_time, save_path)

if __name__ == '__main__':
    main()
