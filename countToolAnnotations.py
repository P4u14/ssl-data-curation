import seaborn as sns
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

def plot_tool_counts(tool_counts_over_time, save_path, title):
    # Set the Seaborn theme and context
    sns.set_theme(style="whitegrid")
    sns.set_context("poster")  # You can switch to "paper", "poster", "talk", or "notebook" depending on your needs
    
    percentages = [10 * i for i in range(1, 11)]
    tools = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]
    
    # Initialize the plot
    plt.figure(figsize=(12, 8))
    
    # Plot each tool's counts with a Seaborn color palette
    palette = sns.color_palette("deep")  # You can choose different palettes like "muted", "bright", etc.
    for tool, color in zip(tools, palette):
        counts = [tool_counts.get(tool, 0) for tool_counts in tool_counts_over_time]
        plt.plot(percentages, counts, marker='o', label=tool, color=color)

    # Set y-axis to start at 0
    plt.ylim(0, None)
    
    # Add labels and title
    plt.xlabel('Percentage of images selected')
    plt.ylabel('Total count of each tool')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xticks(percentages)

     # Adjust the layout to add more space for the title and labels
    plt.tight_layout(pad=2.0)  # pad parameter adds padding between the elements and the edges
    
    
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






    # folder = 'random_images_Cholec80/'
    # tool_counts_over_time = []
    
    """ for image_file in image_files:
        images_file_path = os.path.join(folder, image_file)
        tool_totals = process_images_file(images_file_path, tool_annotations_folder)
        tool_counts_over_time.append(tool_totals) """
    
    # print('tool counts cholec80 random: ')
    # print(tool_counts_over_time)

    # save_path = 'plots/random_images_Cholec80_presentation.png'
    # plot_tool_counts(tool_counts_over_time, save_path, 'Random image selection - Tool counts by percentage of images selected')

    # saved_values = [{'Grasper': 7566, 'Bipolar': 651, 'Hook': 8197, 'Scissors': 216, 'Clipper': 431, 'Irrigator': 701, 'SpecimenBag': 807}, {'Grasper': 15014, 'Bipolar': 1260, 'Hook': 16267, 'Scissors': 512, 'Clipper': 854, 'Irrigator': 1391, 'SpecimenBag': 1722}, {'Grasper': 22592, 'Bipolar': 2018, 'Hook': 24384, 'Scissors': 718, 'Clipper': 1253, 'Irrigator': 2106, 'SpecimenBag': 2603}, {'Grasper': 30224, 'Bipolar': 2773, 'Hook': 32365, 'Scissors': 965, 'Clipper': 1768, 'Irrigator': 2771, 'SpecimenBag': 3470}, {'Grasper': 37832, 'Bipolar': 3420, 'Hook': 40562, 'Scissors': 1216, 'Clipper': 2219, 'Irrigator': 3476, 'SpecimenBag': 4198}, {'Grasper': 45355, 'Bipolar': 4045, 'Hook': 48740, 'Scissors': 1453, 'Clipper': 2639, 'Irrigator': 4220, 'SpecimenBag': 5187}, {'Grasper': 52942, 'Bipolar': 4773, 'Hook': 56517, 'Scissors': 1680, 'Clipper': 3094, 'Irrigator': 4890, 'SpecimenBag': 6079}, {'Grasper': 60706, 'Bipolar': 5415, 'Hook': 64921, 'Scissors': 1938, 'Clipper': 3478, 'Irrigator': 5647, 'SpecimenBag': 6857}, {'Grasper': 67975, 'Bipolar': 6157, 'Hook': 72976, 'Scissors': 2183, 'Clipper': 3946, 'Irrigator': 6356, 'SpecimenBag': 7762}, {'Grasper': 75621, 'Bipolar': 6769, 'Hook': 81135, 'Scissors': 2443, 'Clipper': 4399, 'Irrigator': 7007, 'SpecimenBag': 8585}]
    # plot_tool_counts(saved_values, save_path, 'Distribution of tools in Cholec80 - random image selection')






    folder = 'dinoV2_Cholec80_no_test_images/'
    tool_counts_over_time = []
    
    for image_file in image_files:
        images_file_path = os.path.join(folder, image_file)
        tool_totals = process_images_file(images_file_path, tool_annotations_folder)
        tool_counts_over_time.append(tool_totals)

    print('tool counts cholec80 dinoV2: ')
    print(tool_counts_over_time)
    
    save_path = 'plots/dinoV2_Cholec80_no_test_images_presentation.png'
    plot_tool_counts(tool_counts_over_time, save_path, 'Distribution of tools in Cholec80 - DINOv2')

    saved_values = [{'Grasper': 2594, 'Bipolar': 365, 'Hook': 2697, 'Scissors': 31, 'Clipper': 97, 'Irrigator': 497, 'SpecimenBag': 871}, {'Grasper': 8571, 'Bipolar': 994, 'Hook': 9594, 'Scissors': 237, 'Clipper': 476, 'Irrigator': 1143, 'SpecimenBag': 1738}, {'Grasper': 14186, 'Bipolar': 1643, 'Hook': 16880, 'Scissors': 517, 'Clipper': 887, 'Irrigator': 1751, 'SpecimenBag': 2466}, {'Grasper': 20059, 'Bipolar': 2311, 'Hook': 24333, 'Scissors': 742, 'Clipper': 1370, 'Irrigator': 2336, 'SpecimenBag': 3305}, {'Grasper': 26785, 'Bipolar': 2833, 'Hook': 31457, 'Scissors': 889, 'Clipper': 1706, 'Irrigator': 2823, 'SpecimenBag': 4379}, {'Grasper': 33604, 'Bipolar': 3292, 'Hook': 38981, 'Scissors': 1125, 'Clipper': 2064, 'Irrigator': 3322, 'SpecimenBag': 5311}, {'Grasper': 40917, 'Bipolar': 3742, 'Hook': 46880, 'Scissors': 1358, 'Clipper': 2525, 'Irrigator': 3830, 'SpecimenBag': 5877}, {'Grasper': 48912, 'Bipolar': 4365, 'Hook': 54979, 'Scissors': 1575, 'Clipper': 2896, 'Irrigator': 4434, 'SpecimenBag': 6424}, {'Grasper': 56580, 'Bipolar': 5112, 'Hook': 62952, 'Scissors': 1803, 'Clipper': 3341, 'Irrigator': 5150, 'SpecimenBag': 6975}, {'Grasper': 75621, 'Bipolar': 6769, 'Hook': 81135, 'Scissors': 2443, 'Clipper': 4399, 'Irrigator': 7007, 'SpecimenBag': 8585}]




    """ folder = 'endoViT_Cholec80_no_test_images/'
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
    plot_tool_counts(tool_counts_over_time, save_path) """



if __name__ == '__main__':
    main()
