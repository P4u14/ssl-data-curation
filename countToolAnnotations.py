import os
import re
import matplotlib.pyplot as plt

def count_tools_in_annotation(annotation_file, frame_number):
    # Initialize counts for all tools
    tool_counts = {"Grasper": 0, "Bipolar": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "SpecimenBag": 0}
    
    # Open and read the annotation file
    with open(annotation_file, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:  # Skip the header
            parts = line.strip().split()
            if len(parts) == 8:
                frame = int(parts[0])
                # Check if the frame matches
                if frame == frame_number:
                    tool_counts["Grasper"] = int(parts[1])
                    tool_counts["Bipolar"] = int(parts[2])
                    tool_counts["Hook"] = int(parts[3])
                    tool_counts["Scissors"] = int(parts[4])
                    tool_counts["Clipper"] = int(parts[5])
                    tool_counts["Irrigator"] = int(parts[6])
                    tool_counts["SpecimenBag"] = int(parts[7])
                    break
    return tool_counts

def process_images_file(images_file_path, tool_annotations_folder):
    print('Start evaluating ' + str(tool_annotations_folder) + str(images_file_path))
    # Initialize totals for all tools
    tool_totals = {"Grasper": 0, "Bipolar": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "SpecimenBag": 0}

    # Open and read the images file
    with open(images_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            # Extract the last number from the path
            match = re.search(r'/(\d+)\.pt$', line)
            if match:
                last_number = int(match.group(1))
                frame_number = last_number * 25
                # Determine the annotation file
                match = re.search(r'/(\d+)/\d+\.pt$', line)
                if match:
                    number = match.group(1)
                    annotation_file = os.path.join(tool_annotations_folder, f'video{number}-tool.txt')

                    # Check if the annotation file exists and count tools
                    if os.path.exists(annotation_file):
                        tool_counts = count_tools_in_annotation(annotation_file, frame_number)
                        for tool, count in tool_counts.items():
                            tool_totals[tool] += count

    return tool_totals

def plot_tool_counts(tool_counts_over_time, save_path):
    percentages = [10 * i for i in range(1, 11)]
    tools = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]
    
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
    # File paths
    tool_annotations_folder = '/mnt/ceph/tco/TCO-All/SharedDatasets/Cholec80/tool_annotations/'
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






    folder = 'clip_Cholec80_no_test_images/'
    tool_counts_over_time = []
    
    for image_file in image_files:
        images_file_path = os.path.join(folder, image_file)
        tool_totals = process_images_file(images_file_path, tool_annotations_folder)
        tool_counts_over_time.append(tool_totals)
    
    save_path = 'plots/clip_Cholec80_no_test_images.png'
    plot_tool_counts(tool_counts_over_time, save_path)






    folder = 'dinoV2_Cholec80_no_test_images/'
    tool_counts_over_time = []
    
    for image_file in image_files:
        images_file_path = os.path.join(folder, image_file)
        tool_totals = process_images_file(images_file_path, tool_annotations_folder)
        tool_counts_over_time.append(tool_totals)
    
    save_path = 'plots/dinoV2_Cholec80_no_test_images.png'
    plot_tool_counts(tool_counts_over_time, save_path)




    folder = 'endoViT_Cholec80_no_test_images/'
    tool_counts_over_time = []
    
    for image_file in image_files:
        images_file_path = os.path.join(folder, image_file)
        tool_totals = process_images_file(images_file_path, tool_annotations_folder)
        tool_counts_over_time.append(tool_totals)
    
    save_path = 'plots/endoViT_Cholec80_no_test_images.png'
    plot_tool_counts(tool_counts_over_time, save_path)




    folder = 'endoViTAlex_Cholec80_no_test_images/'
    tool_counts_over_time = []
    
    for image_file in image_files:
        images_file_path = os.path.join(folder, image_file)
        tool_totals = process_images_file(images_file_path, tool_annotations_folder)
        tool_counts_over_time.append(tool_totals)
    
    save_path = 'plots/endoViTAlex_Cholec80_no_test_images.png'
    plot_tool_counts(tool_counts_over_time, save_path)



if __name__ == '__main__':
    main()
