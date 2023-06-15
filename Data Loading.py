import os

import pandas as pd
import pydicom


def get_dicom_metadata(patient_folder):
    metadata = []
    for root, dirs, files in os.walk(patient_folder):
        dicom_files = [file for file in files if file.endswith(".dcm")]
        if dicom_files:
            last_dicom_file = os.path.join(root, dicom_files[-1])
            ds = pydicom.dcmread(last_dicom_file)
            # I will increase row later
            row = {
                "PatientID": ds.get("PatientID", "N/A"),
                "StudyDate": ds.get("StudyDate", "N/A"),
                "SeriesDescription": ds.get("SeriesDescription", "N/A"),
                "StudyInstanceUID": ds.get("StudyInstanceUID", "N/A"),
                "SeriesInstanceUID": ds.get("SeriesInstanceUID", "N/A"),
                "ProtocolName": ds.get("ProtocolName", "N/A"),
                "contrast agent": ds.get("ContrastBolusAgent", "NA")
            }
            metadata.append(row)

    return metadata


# folder path
patient_folder = input("Enter the patient folder path: ")

metadata = get_dicom_metadata(patient_folder)

# Create a dataframe from the metadata
df = pd.DataFrame(metadata)

# Display the dataframe
print(df)
