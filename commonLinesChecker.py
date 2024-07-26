import os
from itertools import combinations

def calculate_common_percentage(file1, file2):
    with open(file1, 'r') as f1:
        lines1 = f1.readlines()
    
    with open(file2, 'r') as f2:
        lines2 = f2.readlines()
    
    set1 = set(lines1)
    set2 = set(lines2)
    
    if len(set1) < len(set2):
        smaller_set, larger_set = set1, set2
    else:
        smaller_set, larger_set = set2, set1
    
    common_lines = smaller_set.intersection(larger_set)
    common_percentage = (len(common_lines) / len(smaller_set)) * 100
    
    return common_percentage

def main(directory, output_file):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]
    pairs = combinations(files, 2)
    
    with open(output_file, 'w') as out_file:
        for file1, file2 in pairs:
            common_percentage = calculate_common_percentage(file1, file2)
            out_file.write(f'Common lines percentage between {os.path.basename(file1)} and {os.path.basename(file2)}: {common_percentage:.2f}%\n')

if __name__ == "__main__":

    directory = 'endoViT_Cholec80_no_test_images'  # Directory containing the files
    output_file = 'common-lines-endoViT_Cholec80_no_test_images.txt'
    main(directory, output_file)

    directory = 'endoViT_SurgOmicFeatures_no_test_images'  # Directory containing the files
    output_file = 'common-lines-endoViT_SurgOmicFeatures_no_test_images.txt'
    main(directory, output_file)

    directory = 'endoViTAlex_Cholec80_no_test_images'  # Directory containing the files
    output_file = 'common-lines-endoViTAlex_Cholec80_no_test_images.txt'
    main(directory, output_file)

    directory = 'endoViTAlex_SurgOmicFeatures_no_test_images'  # Directory containing the files
    output_file = 'common-lines-endoViTAlex_SurgOmicFeatures_no_test_images.txt'
    main(directory, output_file)

    directory = 'dinoV2_Cholec80_no_test_images'  # Directory containing the files
    output_file = 'common-lines-dinoV2_Cholec80_no_test_images.txt'
    main(directory, output_file)

    directory = 'dinoV2_SurgOmicFeatures_no_test_images'  # Directory containing the files
    output_file = 'common-lines-dinoV2_SurgOmicFeatures_no_test_images.txt'
    main(directory, output_file)

    directory = 'clip_Cholec80_no_test_images'  # Directory containing the files
    output_file = 'common-lines-clip_Cholec80_no_test_images.txt'
    main(directory, output_file)

    directory = 'dinoV2_SurgOmicFeatures_no_test_images'  # Directory containing the files
    output_file = 'common-lines-dinoV2_SurgOmicFeatures_no_test_images.txt'
    main(directory, output_file)
