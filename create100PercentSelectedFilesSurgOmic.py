import csv

# Open the CSV file and the output file
with open('/mnt/ceph/tco/TCO-All/SharedDatasets/SurgOmicFeatures/FeatureDetection/annotations.csv', mode='r') as csvfile, open('clip_SurgOmicFeatures_no_test_images/100_percent_selected_image_paths.txt', mode='w') as txtfile:
    # Create a CSV reader object
    reader = csv.reader(csvfile)
    
    # Skip the header row
    next(reader)
    
    # Process each row in the CSV
    for row in reader:
        # The first column contains the filename
        filename = row[0]
        # Replace the .png extension with .plt
        new_filename = filename.replace('.png', '.pt')
        # Write the new filename to the output file
        txtfile.write('/' + new_filename + '\n')
