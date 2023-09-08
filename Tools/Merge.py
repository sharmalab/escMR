import pandas as pd
import os


from tqdm import tqdm

# Directory path containing the CSV files
directory = ''

# Define the starting and ending indices for filenames
start_index = 1
end_index = 1000

# List to store the selected DataFrames from each file
dfs = []

# Selected columns to extract
selected_columns = ["BodyPartExamined", "PatientID", "StudyDate", "StudyDescription", "SeriesDescription",
                    "StudyInstanceUID", "SeriesInstanceUID", "ProtocolName",
                    "ContrastBolusAgent", "ScanningSequence", "SequenceVariant",
                    "SliceThickness", "RepetitionTime", "EchoTime", "ImagingFrequency",
                    "MagneticFieldStrength", "SpacingBetweenSlices", "FlipAngle",
                    "SAR", "ImagePositionPatient", "ImageOrientationPatient",
                    "SliceLocation", "PhotometricInterpretation", "PixelSpacing",
                    "MRAcquisitionType", "EchoTrainLength", "ScanOptions",
                    "ScanningSequence", "ImageType", "SequenceName", "PixelBandwidth", "InversionTime"]

# Create a list of filenames
file_list = [f"metadata_{i}.csv" for i in range(start_index, end_index + 1)]

# Number of files to read
num_files = len(file_list)

# Create a tqdm progress bar for the overall task completion
progress_bar = tqdm(total=num_files, desc='Processing files')

# Iterate over each file in the directory
for filename in file_list:
    file_path = os.path.join(directory, filename)
    try:
        # Read the CSV file with all columns
        df = pd.read_csv(file_path, low_memory=False)

        # Filter rows where "BodyPartExamined" is equal to "BRAIN"
        filtered_df = df[df["BodyPartExamined"] == "BRAIN"].copy()

        # Check for missing columns and add them with default values if necessary
        missing_columns = set(selected_columns) - set(filtered_df.columns)
        for col in missing_columns:
            filtered_df[col] = ''

        # Filter merged_df to only include rows with unique SeriesInstanceUID
        filtered_df_unique = filtered_df.drop_duplicates(subset='SeriesInstanceUID', keep='first')

        selected_df = filtered_df_unique.loc[:, selected_columns]
        dfs.append(selected_df)
    except pd.errors.ParserError:
        # Handle the ParserError and continue to the next file
        print(f"Error reading file: {file_path}. Skipping this file.")
        continue
    # Update the progress bar for each processed file
    progress_bar.update(1)

# Close the progress bar
progress_bar.close()

# Concatenate the selected DataFrames into a single DataFrame
merged_df4 = pd.concat(dfs, ignore_index=True)

# Save the merged_df_unique as a CSV file
output_file = os.path.join(directory, 'merged_df4.csv')
merged_df4.to_csv(output_file, index=False)

print(f"Saved merged_df_unique as CSV file: {output_file}")
