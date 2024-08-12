import os
import random

# Define the path to your input file and the directory to save the output files
# input_file = 'clip_Cholec80_no_test_images/100_percent_selected_image_paths.txt'
# output_dir = 'random_images_Cholec80'
input_file = 'clip_SurgOmicFeatures_no_test_images/100_percent_selected_image_paths.txt'
output_dir = 'random_images_SurgOmic'

# Make sure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read the lines from the input file
with open(input_file, 'r') as file:
    lines = file.readlines()

# Calculate and write the percentages
percentages = range(10, 100, 10)

for percent in percentages:
    # Calculate the number of lines to select
    num_lines_to_select = int(len(lines) * (percent / 100))
    
    # Randomly select the lines
    selected_lines = random.sample(lines, num_lines_to_select)
    
    # Write the selected lines to the corresponding output file
    output_file = os.path.join(output_dir, f'{percent}_percent_selected_image_paths.txt')
    with open(output_file, 'w') as file:
        file.writelines(selected_lines)

print("Random selection files have been created successfully.")
