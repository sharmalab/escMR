import os
import shutil
import pydicom
# Prompt the user for the folder path
folder_path = input("Enter the folder path: ")

metadata = get_dicom_metadata(folder_path)
def change_extension_recursive(folder_path, new_extension):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            current_file_path = os.path.join(root, file)
            temp_file_path = os.path.join(root, "temp" + new_extension)

            try:
                # Check if the file can be read as a valid DICOM file
                ds = pydicom.dcmread(current_file_path)
                # If successful, rename the file with the desired extension
                new_file_path = os.path.splitext(current_file_path)[0] + new_extension
                shutil.move(current_file_path, new_file_path)
                print(f"Renamed: {current_file_path} -> {new_file_path}")
            except pydicom.errors.InvalidDicomError:
                # If the file is not a valid DICOM file, rename it with a temporary extension
                shutil.move(current_file_path, temp_file_path)
                print(f"Renamed: {current_file_path} -> {temp_file_path}")

# Example usage

new_extension = ".dcm"  # Replace with the desired extension

change_extension_recursive(folder_path, new_extension)
