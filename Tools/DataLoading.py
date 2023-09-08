import os
import pandas as pd
import pydicom


def get_dicom_metadata(folder_path):
    metadata = []
    for root, dirs, files in os.walk(folder_path):
        dicom_files = [file for file in files if file.endswith(".dcm")]
        if dicom_files:
            dicom_file = os.path.join(root, dicom_files[0])
            ds = pydicom.dcmread(dicom_file)
            row = {
                "PatientID": ds.get("PatientID", "N/A"),
                "StudyDate": ds.get("StudyDate", "N/A"),
                "StudyDescription": ds.get("StudyDescription", "N/A"),
                "SeriesDescription": ds.get("SeriesDescription", "N/A"),
                "StudyInstanceUID": ds.get("StudyInstanceUID", "N/A"),
                "SeriesInstanceUID": ds.get("SeriesInstanceUID", "N/A"),
                "ProtocolName": ds.get("ProtocolName", "N/A"),
                "ContrastBolusAgent": ds.get("ContrastBolusAgent", "N/A"),
                "ScanningSequence": ds.get("ScanningSequence", "N/A"),
                "SequenceVariant": ds.get("SequenceVariant", "N/A"),
                "SliceThickness": ds.get("SliceThickness", "N/A"),
                "RepetitionTime": ds.get("RepetitionTime", "N/A"),
                "EchoTime": ds.get("EchoTime", "N/A"),
                "ImagingFrequency": ds.get("ImagingFrequency", "N/A"),
                "MagneticFieldStrength": ds.get("MagneticFieldStrength", "N/A"),
                "SpacingBetweenSlices": ds.get("SpacingBetweenSlices", "N/A"),
                "FlipAngle": ds.get("FlipAngle", "N/A"),
                "SAR": ds.get("SAR", "N/A"),
                "ImagePositionPatient": ds.get("ImagePositionPatient", "N/A"),
                "ImageOrientationPatient": ds.get("ImageOrientationPatient", "N/A"),
                "SliceLocation": ds.get("SliceLocation", "N/A"),
                "PhotometricInterpretation": ds.get("PhotometricInterpretation", "N/A"),
                "PixelSpacing": ds.get("PixelSpacing", "N/A"),
                "MRAcquisitionType": ds.get("MRAcquisitionType", "N/A"),
                "InversionTime": ds.get("InversionTime", "N/A"),
                "EchoTrainLength": ds.get("EchoTrainLength", "N/A"),
            }

            metadata.append(row)
    return metadata


# Prompt the user for the folder path
folder_path = input("Enter the folder path: ")

metadata = get_dicom_metadata(folder_path)

# Create a dataframe from the metadata
df = pd.DataFrame(metadata)

# Display the dataframe
print(df)
