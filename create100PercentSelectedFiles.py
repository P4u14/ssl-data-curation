import os
import re

def remove_leading_zeros(filename):
    # Use regex to remove leading zeros from filenames
    return re.sub(r'\b0+(\d)', r'\1', filename)

def collect_image_paths(root_folder, output_file, excluded_dirs):
    # List to hold the formatted image paths
    image_paths = []

    # Convert excluded_dirs to a set for faster lookups
    excluded_dirs_set = set(excluded_dirs)

    # Walk through all subfolders and files
    for subdir, _, files in os.walk(root_folder):
        # Extract the relative path of the current subdir
        relative_subdir = os.path.relpath(subdir, root_folder)
        # Split the relative path into components
        path_parts = relative_subdir.split(os.sep)

        # Only check if the second level subfolder (path_parts[1]) is in the excluded list
        if len(path_parts) > 1 and path_parts[1] in excluded_dirs_set:
            continue  # Skip this directory and its files

        for file in files:
            # Check if the file is an image with a .png extension
            if file.lower().endswith('.png'):
                # Build the relative path of the image
                relative_path = os.path.relpath(os.path.join(subdir, file), root_folder)
                # Split the relative path into components
                parts = relative_path.split(os.sep)

                # Remove leading zeros from the filename
                parts[-1] = remove_leading_zeros(parts[-1])
                # Replace .png with .pt in the file name
                parts[-1] = parts[-1].replace('.png', '.pt')

                # Reconstruct the path and add it to the list
                formatted_path = '/'.join(parts)
                image_paths.append(formatted_path)
    
    # Write the collected paths to the output file
    with open(output_file, 'w') as f:
        for path in image_paths:
            f.write(path + '\n')

if __name__ == "__main__":
    # Define the root folder and output file path
    root_folder = '/mnt/ceph/tco/TCO-All/SharedDatasets/Cholec80/frames_1fps/'
    output_file = 'clip_Cholec80_no_test_images/100_percent_selected_image_paths.txt'
    
    # List of directories to exclude (second level subfolder)
    test_directories = ["/3/", "/24/", "/40/", "/16/", "/32/", "/34/", "/7/", "/14/", "/29/", "/8/", "/31/", "/37/", "/15/", "/30/", "/18/", "/11/", "/6/", "/2/", "/33/", "/21/", "/23/", "/25/", "/19/", "/20/"]
    
    # Extract second level subfolder names (strip leading and trailing slashes)
    excluded_dirs = [d.strip('/') for d in test_directories]
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Collect image paths and write to the file
    collect_image_paths(root_folder, output_file, excluded_dirs)
    print(f"Paths to image files have been written to {output_file}")

