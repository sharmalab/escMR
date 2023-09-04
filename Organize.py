### Code to organize the dicom files in a hierarchical manner 
import os
import shutil
import pydicom
from tqdm import tqdm

def organize_dicom_files(parent_folder):
    # Create a dictionary to store DICOM files based on SeriesInstanceUID
    dicom_dict = {}
    
    # Count the total number of DICOM files for the progress bar
    total_files = sum(1 for _ in get_dicom_files(parent_folder))
    
    # Initialize tqdm progress bar
    progress_bar = tqdm(total=total_files, desc="Organizing DICOM Files")
    
    # Iterate through all DICOM files
    for file_path, series_uid in get_dicom_files(parent_folder):
        if series_uid not in dicom_dict:
            dicom_dict[series_uid] = []
        
        dicom_dict[series_uid].append(file_path)
        progress_bar.update(1)
    
    progress_bar.close()
    
    # Create subfolders and move DICOM files based on SeriesInstanceUID
    for series_uid, file_paths in dicom_dict.items():
        series_folder = os.path.join(parent_folder, series_uid)
        os.makedirs(series_folder, exist_ok=True)
        
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            destination_path = os.path.join(series_folder, file_name)
            shutil.move(file_path, destination_path)
    
    print("DICOM files organized successfully!")

def get_dicom_files(parent_folder):
    for root, _, files in os.walk(parent_folder):
        for file in files:
            if file.lower().endswith('.dcm'):
                file_path = os.path.join(root, file)
                ds = pydicom.dcmread(file_path)
                series_uid = str(ds.SeriesInstanceUID)
                yield file_path, series_uid

# Provide the parent folder path where DICOM files are located
parent_folder_path = input("Enter the folder path: ")

# Call the function to organize DICOM files with tqdm progress bar
organize_dicom_files(parent_folder_path)
