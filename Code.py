from tqdm import tqdm
import re
import joblib
import pydicom
import os
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import pydicom.multival
pd.set_option('display.max_columns', None)
pd.options.display.max_seq_items = 1500
pd.options.display.max_rows = 1500
# Provide the parent folder path where DICOM files are located
folder_path = input("Enter the folder path: ")
change_ext = input("Do you want to change the extension of files to .dcm? Write 'Y' for yes:")
if change_ext == 'Y'  :
    # Define a list of file names to delete
    file_names_to_delete = ["SECTRA", "DICOMDIR", "README.TXT", "CONTENT.XML"]
    def change_extension_recursive(folder_path, new_extension):
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                current_file_path = os.path.join(root, file)

                # Check if the file name is in the list of file names to delete
                if file in file_names_to_delete:
                    # Delete the file using os.remove()
                    os.remove(current_file_path)

                else:
                    try:
                        # Check if the file can be read as a valid DICOM file
                        ds = pydicom.dcmread(current_file_path)
                        # If successful, rename the file with the desired extension
                        new_file_path = os.path.splitext(current_file_path)[0] + new_extension
                        # Use os.rename or os.replace instead of shutil.move
                        os.rename(current_file_path, new_file_path)  # or os.replace(current_file_path, new_file_path)

                    except pydicom.errors.InvalidDicomError:
                        # If the file is not a valid DICOM file, do nothing
                        pass
                        
    new_extension = ".dcm"  # Replace with the desired extension
    change_extension_recursive(folder_path, new_extension)

def get_dicom_metadata(folder_path):
    metadata = []
    for root, dirs, files in tqdm(os.walk(folder_path)):
        dicom_files = [file for file in files if file.endswith(".dcm")]
        if dicom_files:
            dicom_file = os.path.join(root, dicom_files[0])
            ds = pydicom.dcmread(dicom_file)
            row = {
                "StudyDescription": ds.get("StudyDescription", "N/A"),
                "SeriesDescription": ds.get("SeriesDescription", "N/A"),  # Add this line

                "StudyInstanceUID": ds.get("StudyInstanceUID", "N/A"),
                "SeriesInstanceUID": ds.get("SeriesInstanceUID", "N/A"),

                "ProtocolName": ds.get("ProtocolName", "N/A"),
                "ContrastBolusAgent": ds.get("ContrastBolusAgent", "N/A"),
                "ScanningSequence": ds.get("ScanningSequence", "N/A"),
                "ScanOptions": ds.get("ScanOptions", "N/A"),
                "SequenceVariant": ds.get("SequenceVariant", "N/A"),
                "SliceThickness": ds.get("SliceThickness", "N/A"),
                "RepetitionTime": ds.get("RepetitionTime", "N/A"),
                "EchoTime": ds.get("EchoTime", "N/A"),
                "ImagingFrequency": ds.get("ImagingFrequency", "N/A"),
                "MagneticFieldStrength": ds.get("MagneticFieldStrength", "N/A"),
                "SpacingBetweenSlices": ds.get("SpacingBetweenSlices", 'N/A'),
                "ImageType": ds.get("ImageType", "N/A"),
                "FlipAngle": ds.get("FlipAngle", "N/A"),
                "SAR": ds.get("SAR", "N/A"),
                "PercentFOV": ds.get("PercentPhaseFieldOfView", "N/A"),
                "ImagePositionPatient": ds.get("ImagePositionPatient", "N/A"),
                "ImageOrientationPatient": ds.get("ImageOrientationPatient", "N/A"),
                "SliceLocation": ds.get("SliceLocation", "N/A"),
                "PhotometricInterpretation": ds.get("PhotometricInterpretation", "N/A"),
                "PixelSpacing": ds.get("PixelSpacing", "N/A"),
                "MRAcquisitionType": ds.get("MRAcquisitionType", "N/A"),
                "InversionTime": ds.get("InversionTime", "N/A"),
                "EchoTrainLength": ds.get("EchoTrainLength", "N/A"),
                "Rows": ds.get("Rows", "N/A"),
                "Cols": ds.get("Columns", "N/A"),
                "ReconstructionDiameter": ds.get("ReconstructionDiameter", "N/A"),
                "AcquisitionMatrix": ds.get("AcquisitionMatrix", "N/A"),
                "ImagesInAcquisition": ds.get("ImagesInAcquisition", "N/A")
            }

            metadata.append(row)
    return metadata

metadata = get_dicom_metadata(folder_path)

# Create a dataframe from the metadata
df = pd.DataFrame(metadata)
df = df.drop_duplicates(subset='SeriesInstanceUID')
df['SeriesDescription'] = df['SeriesDescription'].str.upper()
df['StudyDescription'] = df['StudyDescription'].str.upper()

# Reset the index to avoid any potential issues with duplicate indices
df.reset_index(drop=True, inplace=True)

df.replace(['NaN', 'N/A'], np.nan, inplace=True)
df['ScanOptions'].fillna('dukh', inplace=True)
df['SequenceVariant'].fillna('dard', inplace=True)
df['ScanningSequence'].fillna('peeda', inplace=True)
df['ImageType'].fillna('kasht', inplace=True)
df['ScanOptions'].fillna('avsaad', inplace=True)

def file_plane(IOP):
    if isinstance(IOP, str):
        IOP_values = [float(x.strip().strip('"')) if x != 'NaN' else math.nan for x in IOP.strip('()').split(',')]
    elif isinstance(IOP, pydicom.multival.MultiValue):
        IOP_values = [float(x) if x != 'NaN' else math.nan for x in IOP]
    else:
        IOP_values = [float(IOP) if not math.isnan(IOP) else math.nan]
    IOP_round = [round(x) if not math.isnan(x) else 0 for x in IOP_values]
    if len(IOP_round) < 6:
        return 'Unknown'
    plane = np.cross(IOP_round[:3], IOP_round[3:])
    plane = [abs(x) for x in plane]
    if plane[0] == 1:
        return 'SAG'
    elif plane[1] == 1:
        return 'COR'
    elif plane[2] == 1:
        return 'AX'
    else:
        return 'Unknown'


def add_anatomical_plane_column(df):
    # Create a copy of the DataFrame
    df_copy = df.copy()

    # Iterate over each row in the DataFrame
    anatomical_planes = []
    for index, row in df_copy.iterrows():
        IOP = row['ImageOrientationPatient']
        anatomical_plane = file_plane(IOP)
        anatomical_planes.append(anatomical_plane)

    # Add the anatomical plane column to the copied DataFrame
    df_copy['Anatomical Plane'] = anatomical_planes

    return df_copy

# Assuming you have a DataFrame called 'df' containing DICOM metadata
df = add_anatomical_plane_column(df)

# Instantiate a MultiLabelBinarizer object
mlb = MultiLabelBinarizer()

# Fit and transform the ImageType column to get binary features
binary_features = mlb.fit_transform(df['ScanOptions'])

# Create a new dataframe with the binary features and the original column names
binary_df = pd.DataFrame(binary_features, columns=mlb.classes_)

# Print the binary dataframe
df.reset_index(drop=True, inplace=True)
binary_df.reset_index(drop=True, inplace=True)

df = pd.concat([df, binary_df], axis=1)


df['ImageType'] = df['ImageType'].astype(str)

# Convert string representation to actual lists
df['ImageType'] = df['ImageType'].apply(eval)

# Get all unique feature values
all_features = set()
for features in df['ImageType']:
    if isinstance(features, list):
        all_features.update(features)

# Create binary features based on unique values
for feature in all_features:
    df[feature] = df['ImageType'].apply(lambda x: int(feature in x) if isinstance(x, list) else 0)


# Convert 'scanningsequence' column to uppercase
df['ScanningSequence'] = df['ScanningSequence'].str.upper()

# Tokenize the 'ScanningSequence' column
selected_column = df['ScanningSequence']
tokenized_column = selected_column.str.split(r'[\s()\[\]]+')

# Flatten the series of lists into a single list
tokenized_column_flat = tokenized_column.explode()

# Get unique tokens
unique_tokens = tokenized_column_flat.unique()

# Create new columns in the DataFrame using unique_tokens as column names
for token in unique_tokens:
    # Check if the token is a valid string
    if isinstance(token, str):
        # Remove quotation marks from the token
        cleaned_token = token.replace("'", "").replace('"', '')
        # Check if the column name already exists, if not, create a new column with it
        if cleaned_token not in df.columns:
            df[cleaned_token] = 0

# Set values to 1 for each row where the token appears in the 'ImageType' column
for idx, tokens in enumerate(tokenized_column):
    # Check if 'tokens' is not NaN (not a float)
    if isinstance(tokens, list):
        for token in tokens:
            # Check if the token is a valid string
            if isinstance(token, str):
                # Remove quotation marks from the token
                cleaned_token = token.replace("'", "").replace('"', '')
                df.at[idx, cleaned_token] = 1

# Convert 'SequenceVariant' column to uppercase
df['SequenceVariant'] = df['SequenceVariant'].str.upper()

# Tokenize the 'SequenceVariant' column
selected_column = df['SequenceVariant']
tokenized_column = selected_column.str.split(r'[\s()\[\]]+')

# Flatten the series of lists into a single list
tokenized_column_flat = tokenized_column.explode()

# Get unique tokens
unique_tokens = tokenized_column_flat.unique()

# Create new columns in the DataFrame using unique_tokens as column names
for token in unique_tokens:
    # Check if the token is a valid string
    if isinstance(token, str):
        # Remove quotation marks from the token
        cleaned_token = token.replace("'", "").replace('"', '')
        # Check if the column name already exists, if not, create a new column with it
        if cleaned_token not in df.columns:
            df[cleaned_token] = 0

# Set values to 1 for each row where the token appears in the 'ImageType' column
for idx, tokens in enumerate(tokenized_column):
    # Check if 'tokens' is not NaN (not a float)
    if isinstance(tokens, list):
        for token in tokens:
            # Check if the token is a valid string
            if isinstance(token, str):
                # Remove quotation marks from the token
                cleaned_token = token.replace("'", "").replace('"', '')
                df.at[idx, cleaned_token] = 1
#copying in PixelSpacingCO so, that can be later used in calculating FOV
df['PixelSpacingCO'] = df['PixelSpacing']
df['ContrastBolusAgent'] = df['ContrastBolusAgent'].apply(lambda x: 1 if isinstance(x, str) and x.strip() else 0)
# If you want to replace NaN with 0 before applying the above operation, use 'fillna':
df['ContrastBolusAgent'] = df['ContrastBolusAgent'].fillna(0).apply(lambda x: 1 if x == 1 else 0)

one_hot_encoded = pd.get_dummies(df['MagneticFieldStrength'])
# Concatenate the one-hot encoded DataFrame with the original DataFrame
df = pd.concat([df, one_hot_encoded], axis=1)

one_hot_encoded = pd.get_dummies(df['PhotometricInterpretation'])

# Concatenate the one-hot encoded DataFrame with the original DataFrame
df = pd.concat([df, one_hot_encoded], axis=1)

one_hot_encoded = pd.get_dummies(df['MRAcquisitionType'])

# Concatenate the one-hot encoded DataFrame with the original DataFrame
df = pd.concat([df, one_hot_encoded], axis=1)
def extract_pixel_spacing(string):
    """Extract the numeric part of a string representing pixel spacing."""
    match = re.search(r'\d+\.*\d*', str(string))
    if match:
        return float(match.group())
    else:
        return np.nan
#getting first value
df['PixelSpacing'] = df['PixelSpacing'].apply(extract_pixel_spacing)

df = df.dropna(subset=['PixelSpacing'])

df['InversionTime'] = df['InversionTime'].fillna(0)
# so later while doing compliance check can check for NaN
kk = df[['SpacingBetweenSlices','SliceThickness']]
df.fillna(100000, inplace=True)
# These columns should be present for RF model, it will create the required column if are not present
filtered_rows = [
    'ContrastBolusAgent',
    'SliceThickness', 'RepetitionTime', 'EchoTime', 'ImagingFrequency',
    'SpacingBetweenSlices', 'FlipAngle', 'SAR',
    'SliceLocation',
    'PixelSpacing',
    'EchoTrainLength',
    'PixelBandwidth', 'InversionTime',
    'DERIVED',
    'PRIMARY', 'DIFFUSION', 'TRACEW', 'ND', 'NORM', 'ADC', 'ORIGINAL', 'M',
    'DIS2D', 'SH1_1', 'FIL', 'FM3_2', 'MPR', 'OTHER', 'DIS3D', 'MFSPLIT',
    'MIP_SAG', 'MIP_COR', 'MIP_TRA', 'SECONDARY', 'PROJECTIONIMAGE',
    'CSAMIP', 'CSAMANIPULATED', 'CSAPARALLEL', 'SUB', 'SH', 'NONE', 'FA',
    'TENSOR_B0', 'FM', 'CSAMPRTHICK', '', 'CSAMPR', 'MOSAIC', 'R',
    'PERFUSION', 'REFORMATTED', 'AVERAGE', 'PJN', 'MIP', 'MOCO_ADV', 'MSUM',
    'RD', 'POSDISP', 'CSARESAMPLED', 'CPR', 'DIXON', 'WATER', 'STDDEV_SAG',
    'STDDEV_COR', 'STDDEV_TRA', 'CPR_STAR', 'COLLAPSE', 'COMP_SP',
    'COMPOSED', 'PFP', 'FS', 'IR', 'dukh', 'SAT1', 'EDR_GEMS',
    'FILTERED_GEMS', 'ACC_GEMS', 'PFF', 'EPI_GEMS', 'FAST_GEMS',
    'FC_SLICE_AX_GEMS', 'FC', 'TRF_GEMS', 'FSL_GEMS', 'T2FLAIR_GEMS',
    'SAT_GEMS', 'FR_GEMS', 'CG', 'RG', 'PER', 'SP', 'SFS', 'FSA_GEMS',
    'FSI_GEMS', 'SEQ_GEMS', 'T1FLAIR_GEMS', 'IR_GEMS', 'MP_GEMS', 'SS_GEMS',
    'NPW', 'NP', 'FSS_GEMS', 'CL_GEMS', 'ARTM_GEMS', 'FC_FREQ_AX_GEMS',
    'IFLOW_GEMS', 'SAT2', 'FSP_GEMS', 'HYPERSENSE_GEMS', 'FLEX_GEMS',
    'SAT3', 'VASCTOF_GEMS', 'EP', 'GR,', 'SE', 'SE,', 'GR', 'RM', 'EP,',
    'RM,', 'SK,', 'SP,', 'MP,', 'OSP', 'MP', 'SK', 'SS', 'SS', 1.5, 'MONOCHROME2',
    3.0, '2D', 'MIN IP', 'MNIP', '3D']

missing_columns = [col for col in filtered_rows if col not in df.columns]
missing_df = pd.DataFrame(0, index=df.index, columns=missing_columns)

# Concatenate the original DataFrame and the new DataFrame with missing columns
df = pd.concat([df, missing_df], axis=1)
# Load the trained model from the file
clf = joblib.load('RandomForestLocal.pkl')

final_df = df[[
    'ContrastBolusAgent',
    'SliceThickness', 'RepetitionTime', 'EchoTime', 'ImagingFrequency',
    'SpacingBetweenSlices', 'FlipAngle', 'SAR',
    'SliceLocation',
    'PixelSpacing',
    'EchoTrainLength',
    'PixelBandwidth', 'InversionTime',
    'DERIVED',
    'PRIMARY', 'DIFFUSION', 'TRACEW', 'ND', 'NORM', 'ADC', 'ORIGINAL', 'M',
    'DIS2D', 'SH1_1', 'FIL', 'FM3_2', 'MPR', 'OTHER', 'DIS3D', 'MFSPLIT',
    'MIP_SAG', 'MIP_COR', 'MIP_TRA', 'SECONDARY', 'PROJECTIONIMAGE',
    'CSAMIP', 'CSAMANIPULATED', 'CSAPARALLEL', 'SUB', 'SH', 'NONE', 'FA',
    'TENSOR_B0', 'FM', 'CSAMPRTHICK', '', 'CSAMPR', 'MOSAIC', 'R',
    'PERFUSION', 'REFORMATTED', 'AVERAGE', 'PJN', 'MIP', 'MOCO_ADV', 'MSUM',
    'RD', 'POSDISP', 'CSARESAMPLED', 'CPR', 'DIXON', 'WATER', 'STDDEV_SAG',
    'STDDEV_COR', 'STDDEV_TRA', 'CPR_STAR', 'COLLAPSE', 'COMP_SP',
    'COMPOSED', 'PFP', 'FS', 'IR', 'dukh', 'SAT1', 'EDR_GEMS',
    'FILTERED_GEMS', 'ACC_GEMS', 'PFF', 'EPI_GEMS', 'FAST_GEMS',
    'FC_SLICE_AX_GEMS', 'FC', 'TRF_GEMS', 'FSL_GEMS', 'T2FLAIR_GEMS',
    'SAT_GEMS', 'FR_GEMS', 'CG', 'RG', 'PER', 'SP', 'SFS', 'FSA_GEMS',
    'FSI_GEMS', 'SEQ_GEMS', 'T1FLAIR_GEMS', 'IR_GEMS', 'MP_GEMS', 'SS_GEMS',
    'NPW', 'NP', 'FSS_GEMS', 'CL_GEMS', 'ARTM_GEMS', 'FC_FREQ_AX_GEMS',
    'IFLOW_GEMS', 'SAT2', 'FSP_GEMS', 'HYPERSENSE_GEMS', 'FLEX_GEMS',
    'SAT3', 'VASCTOF_GEMS', 'EP', 'GR,', 'SE', 'SE,', 'GR', 'RM', 'EP,',
    'RM,', 'SK,', 'SP,', 'MP,', 'OSP', 'MP', 'SK']]

predictions = clf.predict(final_df)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
original_labels = ['DWI', 'T1_MPRAGE', 'T2', 'FLAIR', 'T2*', 'T1', 'SCOUT', 'VIBE',
                   'CISS', 'TOF', 'DIR_SPACE', 'T2_SPACE', 'PERF', 'DTI', 'FGATIR',
                   'T1_FLAIR', 'MRV', 'FIESTA', 'MIP', 'MRA']

# Fit and transform the target variable 'y' to numerical values
y_encoded = label_encoder.fit(original_labels)
original_class_names = label_encoder.inverse_transform(predictions)
# Transform the original labels to encoded labels

df['Sequencename'] = original_class_names

mask = (df['Sequencename'] == 'T2*') & (df['MRAcquisitionType'] == '3D')
df.loc[mask, 'Sequencename'] = 'SWI'

df['Probability'] = np.max(clf.predict_proba(final_df), axis=1)
predicted_probabilities = clf.predict_proba(final_df)

sorted_probabilities = np.sort(predicted_probabilities[: ])[:,-2]
df['SecondHighestProbability'] = sorted_probabilities
def calculate_fov(df):
    # Initialize lists to store calculated FOV values
    fov_x_values = []
    fov_y_values = []

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Extract pixel spacing, rows, and columns for the current row
        pixel_spacing = row['PixelSpacingCO']
        rows = row['Rows']
        columns = row['Cols']

        # Calculate FOV for the current row
        fov_x = columns * pixel_spacing[0]
        fov_y = rows * pixel_spacing[1]

        # Append calculated FOV values to the lists
        fov_x_values.append(fov_x)
        fov_y_values.append(fov_y)
        fov = max(fov_x_values, fov_y_values)
    # Add the calculated FOV values to the DataFrame
    df['FOVx'] = fov_x_values
    df['FOVy'] = fov_y_values
    df['FOV'] = fov
    return df

# Call the function to calculate FOV for each row in the DataFrame
df = calculate_fov(df)
def calculate_slice_gap(df):
    if "SpacingBetweenSlices" not in df.columns or "SliceThickness" not in df.columns:
        raise ValueError("Required columns 'SpacingBetweenSlices' and 'SliceThickness' not found in DataFrame.")

    # Convert columns to numeric if they contain string representations
    df["SpacingBetweenSlices"] = pd.to_numeric(df["SpacingBetweenSlices"], errors="coerce")
    df["SliceThickness"] = pd.to_numeric(df["SliceThickness"], errors="coerce")

    df["SliceGap"] = df["SpacingBetweenSlices"] - df["SliceThickness"]
    df["SliceGap"] = df["SliceGap"].abs()
    return df
calculate_slice_gap(df)
def calculate_pixel_area(df):
    # Calculate the width of each pixel in the y-direction
    pixel_width_y = df['FOVy'] / df['Rows']
    pixel_width_x = df['FOVx'] / df['Cols']
    # Assuming square pixels, the area of each pixel is pixel_width_y squared
    pixel_area = pixel_width_y * pixel_width_x
    df['pixel_area'] = pixel_area
    return df
calculate_pixel_area(df)

def calculate_volumetric_coverage(df):
    # Calculate the volumetric coverage by multiplying the number of slices with the spacing between slices
    vol_cov = df['ImagesInAcquisition'] * (df['SpacingBetweenSlices'] + df['SliceThickness'])

    df['vol_cov'] = vol_cov
    return df
calculate_volumetric_coverage(df)

def generate_sequence(row):
    sequence = 'ADC' if row['ADC'] == 1 else row['Sequencename']

    if row['FS'] == 1:
        sequence += 'FS'

    if row['ContrastBolusAgent'] == 1:
        sequence += 'POST'

    if row['MPR'] :
        sequence += 'MPR'

    if row['MIP'] == 1 or row['MIN IP'] == 1 or row['MNIP'] == 1:
        sequence += 'MIP'

    sequence +=row['MRAcquisitionType'] + row['Anatomical Plane']
    return sequence


# Apply the function to each row to create the 'sequence' column
df['sequence'] = df.apply(generate_sequence, axis=1)

criteria = {'B1A': {
    'PRESENCE': {'DWIFS2DAX',
                 'T12DAX',
                 'SWI3DAX',
                 'SWIMIP3DAX',
                 'FLAIR2DAX',
                 'T2*2DAX',
                 'T2FPOST2DAX',
                 'T12DSAG',
                 'ADCFS2DAX', },
    'LENGTH': {8, 9}

},
    'B2A': {
        'PRESENCE': {'DWIFS2DAX',
                     'FLAIR2DAX',
                     'T12DSAG',
                     'SWIMIP3DAX',
                     'T12DAX',
                     'T2FSPOST2DAX',
                     'T1POST2DAX',
                     'T1POST3DSAG',
                     'SWI3DAX',
                     'SWIMIP3DAX',
                     'ADCFS2DAX',
                     'T1MPRAGEMPR3DCOR',
                     'T1MPRAGEMPR3DAX'},
        'LENGTH': {12, 13}
        # Add more sequences and criteria here
    },

    'B2B': {
        'PRESENCE': {'DWIFS2DAX',
                     'FLAIR2DAX',
                     'T12DSAG',
                     'T2*2DAX',
                     'T12DAX',
                     'T2FSPOST2DAX',
                     'T1POST2DAX',
                     'T1MPRAGEPOSTAX',
                     'PERFAX',
                     'SWI3DAX',
                     'SWIMIP3DAX'
            , 'ADCFS2DAX',
                     'T1MPRAGEMPR3DCOR',
                     'T1MPRAGEMPR3DAX'},
        'LENGTH': {12, 13}
        # Add more sequences and criteria here
    },
   

    'B3A': {
        'PRESENCE': {'DWIFS2DAX',
                     'FLAIR3DSAG',
                     'T1MPRAGE3DSAG',
                     'T2*2DAX',
                     'T12DAX',
                     'T2FSPOST2DAX',
                     'T1POST2DAX',
                     'T1MPRAGEPOST3DSAG',
                     'SWI3DAX',
                     'SWIMIP3DAX'
            , 'ADCFS2DAX',
                     'T1MPRAGEMPR3DCOR',
                     'T1MPRAGEMPR3DAX',
                     'FLAIRMPR3DAX',
                     'FLAIRMPR3DCOR'

                     # Add more sequences and criteria here
                     },
        'LENGTH': {13, 14}
    },
    'DWIFS2DAX': {
        'Orientation': '2D',
        'FOV': [220, 250],
        'PixelArea': 3,
        'Thickness': 5,
        'Gap': 1,
        'Coverage': 160
    },
    'DWI2DCOR': {
        'Orientation': '2D',
        'FOV': [220, 240],
        'PixelArea': 3,
        'Thickness': 5,
        'Gap': 1,
        'Coverage': 180
    },
    'SWI3DAX': {
        'Orientation': '3D',
        'FOV': [220, 250],
        'PixelArea': 1.2,
        'Thickness': 4,
        'Gap': 0,
        'Coverage': 160
    },
    'FLAIR2DAX': {
        'Orientation': '2D',
        'FOV': [220, 250],
        'PixelArea': 1.2,
        'Thickness': 5,
        'Gap': 1,
        'Coverage': 160
    },

    'T2*2DAX': {
        'Orientation': '2D',
        'FOV': [220, 250],
        'PixelArea': 1.2,
        'Thickness': 5,
        'Gap': 1,
        'Coverage': 160
    },

    'T12DAX': {
        'Orientation': '2D',
        'FOV': [220, 250],
        'PixelArea': 1.2,
        'Thickness': 5,
        'Gap': 1,
        'Coverage': 160
    },
    'T1POST2DAX': {
        'Orientation': '2D',
        'FOV': [220, 250],
        'PixelArea': 1.2,
        'Thickness': 5,
        'Gap': 1,
        'Coverage': 160
    },

    'T2FSPOST2DAX': {
        'Orientation': '2D',
        'FOV': [220, 250],
        'PixelArea': 1.2,
        'Thickness': 5,
        'Gap': 1,
        'Coverage': 160
    },
    'T12DSAG': {
        'Orientation': '2D',
        'FOV': [230, 260],
        'PixelArea': 1.2,
        'Thickness': 5,
        'Gap': 1,
        'Coverage': 160
    },
    'T1POST2DSAG': {
        'Orientation': '2D',
        'FOV': [140, 180],
        'PixelArea': 0.6,
        'Thickness': 3,
        'Gap': 0.5,
        'Coverage': 50
    },

    'T1POST3DSAG': {
        'Orientation': '3D',
        'FOV': [230, 260],
        'PixelArea': 1.2,
        'Thickness': 1.2,
        'Gap': 0,
        'Coverage': 160
    },





}

df[['SpacingBetweenSlices','SliceThickness']] = kk[['SpacingBetweenSlices','SliceThickness']]

groups = df.groupby("StudyInstanceUID")
num_groups = len(groups)

# Print the number of groups
print("Number of groups present in parent folder:", num_groups)
# For each group, perform compliance checks for each sequence
i = 0
for group_name, group_data in groups:
    print("Processing group:", group_name)
    print("Processing study :",group_data['StudyDescription'][0])
    i = i + 1

    #print(group_data[['StudyDescription', 'SeriesDescription', 'sequence']])
    # Ask the user if they want to go with a default protocol compliance check or custom protocol
    protocol_option = input(
        "Do you want to go with a default protocol compliance check or custom protocol? Enter 'default' or 'custom': ")

    # If the user chooses default protocol, ask for the protocol name and search it in the criteria dictionary
    if protocol_option == "default":
        protocol_name = input("Enter the protocol name: ")
        df_length = len(group_data)
        sequence_names = group_data['sequence'].tolist()
        missing_sequence_names = []

        # Initialize an empty DataFrame to store results
        results_data = []
        data = []
        if protocol_name in criteria.keys():
            presence = criteria[protocol_name]['PRESENCE']
            for sequence_name in presence:
                if sequence_name not in sequence_names:
                    missing_sequence_names.append(sequence_name)


            column_name = ['sequence']
            missing_sequence_names = pd.DataFrame(missing_sequence_names, columns= column_name)

            sk = -len(missing_sequence_names)
            length_range = criteria[protocol_name]['LENGTH']
            if df_length in length_range:
                print("Dataframe length is within range:", df_length)
            else:
                print("Dataframe length is out of range:", df_length, "instead of", length_range)
            duplicate_sequences = group_data[group_data.duplicated(subset='sequence', keep=False)]

            # Get the list of duplicate strings
            duplicate_sequence_list = duplicate_sequences['sequence'].tolist()

            # Print the list of duplicate strings
            print("List of Duplicate sequences in study:")
            print(duplicate_sequence_list)
            for index, row in group_data.iterrows():
                sequence_name = row["sequence"]
                orientation = row["MRAcquisitionType"]
                fov = row["FOV"]
                pixel_area = row["pixel_area"]
                thickness = row["SliceThickness"]
                gap = row["SliceGap"]
                coverage = row["vol_cov"]
                spacing = row['SpacingBetweenSlices']
                score = 1
                # Use code blocks to display formatted code
                if sequence_name in criteria.keys():
                    # Use parentheses to enclose the f-strings
                    orientation_result = 1 if orientation == criteria[sequence_name][
                        "Orientation"] else -1
                    fov_result = 1 if criteria[sequence_name]["FOV"][0] <= fov <= \
                                                    criteria[sequence_name]["FOV"][
                                                        1] else -1
                    pixel_area_result = 1 if pixel_area <= criteria[sequence_name][
                        "PixelArea"] else -1
                    # Use indentation to separate the if-else blocks
                    if math.isnan(thickness):
                        thickness_result = 0
                    else:
                        thickness_result = 1 if thickness <= \
                                                                                    criteria[sequence_name][
                                                                                        "Thickness"] else -1
                    if math.isnan(thickness) or math.isnan(spacing):
                        gap_result = 0
                    else:
                        gap_result = 1 if gap <= criteria[sequence_name][
                            "Gap"] else -1
                    if math.isnan(thickness) or math.isnan(spacing):
                        coverage_result = 0
                    else:
                        coverage_result = 1 if coverage >= \
                                                                                     criteria[sequence_name][
                                                                                         "Coverage"] else -1
                    # Deduct the score by 1 if any of the results are not compliant
                    if orientation_result == -1 or fov_result == -1 or \
                        pixel_area_result == -1 or \
                        thickness_result == -1 or \
                        gap_result == -1 or \
                        coverage_result == -1:
                        score = -1

                    results_data.append(
                        [sequence_name, orientation_result, fov_result, pixel_area_result, thickness_result, gap_result,
                         coverage_result, score])  # Add the score to the results data

                # Convert the results data to a DataFrame with a new column for score
                result_df = pd.DataFrame(results_data,
                                         columns=["Sequence", "Orientation", "FOV", "Pixel Area", "Thickness", "Gap",
                                                  "Coverage", "Score"])


            group_data.reset_index(drop=True, inplace=True)
            missing_sequence_names.reset_index(drop=True, inplace=True)

            # Concatenate the DataFrames vertically
            resultk = pd.concat([group_data[['SeriesDescription', 'sequence','Probability','SecondHighestProbability']], missing_sequence_names], axis=0, ignore_index=True)


            result_df = pd.merge(resultk, result_df, left_on='sequence', right_on='Sequence', how='outer')

            # Optionally, you can drop one of the key columns if you want to keep only one
            result_df.drop(columns=['Sequence'], inplace=True)
            result_df = result_df.drop_duplicates(subset=['sequence'])
            count_minus_1 = (result_df['Score'] == -1).sum()
            #result_df.fillna(0, inplace=True)
            print(result_df)
            final_score = 10
            final_score = final_score - count_minus_1+sk
            print(final_score, "is the study score")
            print("Number of groups remaining :", num_groups - i)
        else:
            print( protocol_name, "is not in the criteria dictionary")
        # Count the number of series in the first group and check it with a given number

    # If the user chooses custom protocol, ask for their own criteria and check them accordingly
    elif protocol_option == "custom":
        # Ask for their own criteria and store them in a dictionary
        #print("write sequence name as shown in sequence column")


        def get_input_sequence_data():
            sequence_data = {}
            sequence_data['Orientation'] = input("Enter Orientation: ")
            sequence_data['FOV'] = [float(x) for x in input("Enter the fov range (e.g. 0.5,0.5): ").split(",")]
            sequence_data['PixelArea'] = float(input("Enter PixelArea: "))
            sequence_data['Thickness'] = float(input("Enter Thickness: "))
            sequence_data['Gap'] = float(input("Enter Gap: "))
            sequence_data['Coverage'] = float(input("Enter Coverage: "))
            return sequence_data


        def create_custom_criteria():
            custom_criteria = {}

            protocol_name = input("Enter custom protocol name: ")
            presence_sequences = input("Enter sequences in PRESENCE (comma-separated): ").split(',')
            length_range = input("Enter LENGTH range (e.g., 12,13): ").split(',')
            length_set = set(range(int(length_range[0]), int(length_range[1]) + 1))

            custom_criteria[protocol_name] = {
                'PRESENCE': set(presence_sequences),
                'LENGTH': length_set
            }

            while True:
                sequence_name = input("Enter sequence name (or 'done' to finish): ")
                if sequence_name.lower() == 'done':
                    break
                sequence_data = get_input_sequence_data()
                custom_criteria[sequence_name] = sequence_data

            return custom_criteria, protocol_name


        # Call the function to create a custom criteria dictionary
        custom_criteria, protocol_name = create_custom_criteria()

        # Print the custom criteria dictionary
        print(custom_criteria)
        df_length = len(group_data)
        sequence_names = group_data['sequence'].tolist()
        missing_sequence_names = []

        # Initialize an empty DataFrame to store results
        results_data = []
        data = []
        if protocol_name in custom_criteria.keys():
            presence = custom_criteria[protocol_name]['PRESENCE']
            for sequence_name in presence:
                if sequence_name not in sequence_names:
                    missing_sequence_names.append(sequence_name)


            column_name = ['sequence']
            missing_sequence_names = pd.DataFrame(missing_sequence_names, columns= column_name)

            sk = -len(missing_sequence_names)
            length_range = custom_criteria[protocol_name]['LENGTH']
            if df_length in length_range:
                print("Dataframe length is within range:", df_length)
            else:
                print("Dataframe length is out of range:", df_length, "instead of", length_range)
            duplicate_sequences = group_data[group_data.duplicated(subset='sequence', keep=False)]

            # Get the list of duplicate strings
            duplicate_sequence_list = duplicate_sequences['sequence'].tolist()

            # Print the list of duplicate strings
            print("List of Duplicate sequences in study:")
            print(duplicate_sequence_list)
            for index, row in group_data.iterrows():
                sequence_name = row["sequence"]
                orientation = row["MRAcquisitionType"]
                fov = row["FOV"]
                pixel_area = row["pixel_area"]
                thickness = row["SliceThickness"]
                gap = row["SliceGap"]
                coverage = row["vol_cov"]
                spacing = row['SpacingBetweenSlices']
                score = 1
                # Use code blocks to display formatted code
                if sequence_name in custom_criteria.keys():
                    # Use parentheses to enclose the f-strings
                    orientation_result = 1 if orientation == custom_criteria[sequence_name][
                        "Orientation"] else -1
                    fov_result = 1 if custom_criteria[sequence_name]["FOV"][0] <= fov <= \
                                                    custom_criteria[sequence_name]["FOV"][
                                                        1] else -1
                    pixel_area_result = 1 if pixel_area <= custom_criteria[sequence_name][
                        "PixelArea"] else -1
                    # Use indentation to separate the if-else blocks
                    if math.isnan(thickness):
                        thickness_result = 0
                    else:
                        thickness_result = 1 if thickness <= \
                                                                                    custom_criteria[sequence_name][
                                                                                        "Thickness"] else -1
                    if math.isnan(thickness) or math.isnan(spacing):
                        gap_result = 0
                    else:
                        gap_result = 1 if gap <= custom_criteria[sequence_name][
                            "Gap"] else -1
                    if math.isnan(thickness) or math.isnan(spacing):
                        coverage_result = 0
                    else:
                        coverage_result = 1 if coverage >= \
                                                                                     custom_criteria[sequence_name][
                                                                                         "Coverage"] else -1
                    # Deduct the score by 1 if any of the results are not compliant
                    if orientation_result == -1 or fov_result == -1 or \
                        pixel_area_result == -1 or \
                        thickness_result == -1 or \
                        gap_result == -1 or \
                        coverage_result == -1:
                        score = -1

                    results_data.append(
                        [sequence_name, orientation_result, fov_result, pixel_area_result, thickness_result, gap_result,
                         coverage_result, score])  # Add the score to the results data

                # Convert the results data to a DataFrame with a new column for score
                result_df = pd.DataFrame(results_data,
                                         columns=["Sequence", "Orientation", "FOV", "Pixel Area", "Thickness", "Gap",
                                                  "Coverage", "Score"])


            group_data.reset_index(drop=True, inplace=True)
            missing_sequence_names.reset_index(drop=True, inplace=True)

            # Concatenate the DataFrames vertically
            resultk = pd.concat([group_data[['SeriesDescription', 'sequence','Probability','SecondHighestProbability']], missing_sequence_names], axis=0, ignore_index=True)


            result_df = pd.merge(resultk, result_df, left_on='sequence', right_on='Sequence', how='outer')

            # Optionally, you can drop one of the key columns if you want to keep only one
            result_df.drop(columns=['Sequence'], inplace=True)
            result_df = result_df.drop_duplicates(subset=['sequence'])
            count_minus_1 = (result_df['Score'] == -1).sum()
            result_df.fillna(0, inplace= True)
            print(result_df)
            final_score = 10
            final_score = final_score - count_minus_1+sk
            print(final_score, "is the study score")
            print("Number of groups remaining :", num_groups - i)
        else:
            print( protocol_name, "is not in the criteria dictionary")
        # Count the number of series in the first group and check it with a given number


    # If the user enters an invalid option, show an error message and exit
    if protocol_option != 'custom' and protocol_option != 'default':
        print("Invalid option. Please enter 'default' or 'custom'.")



print("Exiting the program.")
exit()
