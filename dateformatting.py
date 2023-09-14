import os
import pandas as pd


root_dir = r"C:\Cycling Data/0002"

# Iterate over subfolders
for foldername in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, foldername)

    # Check if it is a directory
    if os.path.isdir(folder_path):
        # Iterate over files in the subfolder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)


            if filename.endswith(".csv") and ("HR_modified" in filename or "WL_modified" in filename):
                # Reading the CSV file
                df = pd.read_csv(file_path)


                # merging heart rate and workload data
                if "HR_modified" in filename:
                    heart_rate_data = df
                    workload_file = os.path.join(folder_path, filename.replace("HR_modified", "WL_modified"))
                    if os.path.isfile(workload_file):
                        workload_data = pd.read_csv(workload_file)
                        merged_data = pd.merge(heart_rate_data, workload_data, on='Time')
                        # Save the merged data to a new CSV file
                        output_path = os.path.join(folder_path, "merged_data.csv")
                        merged_data.to_csv(output_path, index=False)
