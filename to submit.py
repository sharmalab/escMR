#!/usr/bin/env python
# coding: utf-8




from tqdm import tqdm
import re
import joblib
import pydicom
import os
import shutil
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

                "ImageType": ds.get("ImageType", "N/A"), "ProtocolName": ds.get("ProtocolName", "N/A"),
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


# Prompt the user for the folder path

metadata = get_dicom_metadata(folder_path)

# Create a dataframe from the metadata
df = pd.DataFrame(metadata)

df = df.drop_duplicates(subset='SeriesInstanceUID')

# empty_rows = df[df['SeriesDescription'].isna() | df['SeriesDescription'].isnull()]
# df = df.drop(empty_rows.index)


df['SeriesDescription'] = df['SeriesDescription'].str.upper()

df['StudyDescription'] = df['StudyDescription'].str.upper()



# Assuming 'df' is the original DataFrame containing the 'StudyDescription' column
# Reset the index to avoid any potential issues with duplicate indices
df.reset_index(drop=True, inplace=True)
# C:\Users\HIMANSHU GUPTA\Downloads\20230524_18824658_b2a\20230524_18824658_b2a
# C:\Users\HIMANSHU GUPTA\Downloads\Emory data\20230503_11210021_b3a
df.replace(['NaN', 'N/A'], np.nan, inplace=True)
df['ScanOptions'].fillna('dukh', inplace=True)
df['SequenceVariant'].fillna('dard', inplace=True)
df['ScanningSequence'].fillna('peeda', inplace=True)
df['ImageType'].fillna('kasht', inplace=True)
df['ScanOptions'].fillna('avsaad', inplace=True)
kk = df





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





### working with local


# Import pandas and MultiLabelBinarizer


# Create a sample dataframe with ImageType column


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






# working for local image type
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
df['PixelSpacingCO'] = df['PixelSpacing']


def extract_pixel_spacing(string):
    """Extract the numeric part of a string representing pixel spacing."""
    match = re.search(r'\d+\.*\d*', str(string))
    if match:
        return float(match.group())
    else:
        return np.nan


df['PixelSpacing'] = df['PixelSpacing'].apply(extract_pixel_spacing)
df = df.dropna(subset=['PixelSpacing'])

df['InversionTime'] = df['InversionTime'].fillna(0)
df.fillna(100000, inplace=True)




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
result_df = pd.concat([df, missing_df], axis=1)
df = pd.concat([df, missing_df], axis=1)

# In[12]:




# Load the trained model from the file
clf = joblib.load('RandomForestLocal.pkl')




final_df = result_df[[
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

# In[14]:


predictions = clf.predict(final_df)


# array([ 3, 14, 15,  6, 16, 12, 11, 19,  0, 18,  1, 17, 10,  2,  4, 13,  9,
#         5,  7,  8])
# array(['DWI', 'T1_MPRAGE', 'T2', 'FLAIR', 'T2*', 'T1', 'SCOUT', 'VIBE',
#        'CISS', 'TOF', 'DIR_SPACE', 'T2_SPACE', 'PERF', 'DTI', 'FGATIR',
#        'T1_FLAIR', 'MRV', 'FIESTA', 'MIP', 'MRA'], dtype=object)



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




# Define a function to generate the sequence string based on the row values

def generate_sequence(row):
    sequence = 'ADC' if row['ADC'] == 1 else row['Sequencename']

    if row['FS'] == 1:
        sequence += 'FS'

    if row['ContrastBolusAgent'] == 1:
        sequence += 'POST'

    if row['MPR'] == 1:
        sequence += 'MPR'

    if row['MIP'] == 1 or row['MIN IP'] == 1 or row['MNIP'] == 1:
        sequence += 'MIP'

    sequence += row['MRAcquisitionType'] + row['Anatomical Plane']
    return sequence


# Apply the function to each row to create the 'sequence' column
df['sequence'] = df.apply(generate_sequence, axis=1)


# as pixel spacing is mostly present there is no need for using reconstruction diameter, as we are not going to check compliance of derived images


# I can name sequence whatever in criteria and sequencename, so if  i don't get T1_MPRAGE but if I can identify  by other things like T1post3dax is actually t1mpragepost3dax then I can go with it also.
# similar case with MPR , as user will see the sequence name so, i can tell him only to input showing sequence names.
# use SUB and PERFUSION.
# No need to worry if name doesn't come as wanted, as user can still adjust name in his/her custom criteria file.




criteria = {'B1A': {
    'PRESENCE': {'DWIAX',
                 'T1AX',
                 'SWIAX',
                 'SWIMIPAX',
                 'FLAIRAX',
                 'T2*AX',
                 'T2FSAX',
                 'T1SAG',
                 'ADCAX', },
    'LENGTH': {8, 9}

},
    'B2A': {
        'PRESENCE': {'DWI2DAX',
                     'FLAIR2DAX',
                     'T12DSAG',
                     'SWIMIP3DAX',
                     'T12DAX',
                     'T2FSPOST2DAX',
                     'T1POST2DAX',
                     'T1POST3DSAG',
                     'SWI3DAX',
                     'SWIMIP3DAX',
                     'ADC2DAX',
                     'T1POST3DCOR',
                     'T1POST3DAX'},
        'LENGTH': {12, 13}
        # Add more sequences and criteria here
    },

    'B2B': {
        'PRESENCE': {'DWIAX',
                     'FLAIRAX',
                     'T1SAG',
                     'T2*',
                     'T1AX',
                     'T2FSPOSTAX',
                     'T1POSTAX',
                     'T1MPRAGEPOSTAX',
                     'PERFUSIONAX',
                     'SWIAX',
                     'SWIMINIPAX'
            , 'ADCAX',
                     'T1MPRAGEMPRCOR',
                     'T1MPRAGEMPRAX'},
        'LENGTH': {12, 13}
        # Add more sequences and criteria here
    },
    'B2C': {
        'PRESENCE': {'DWIAX',
                     'FLAIRAX',
                     'T1SAG',
                     'T2*AX',
                     'T1AX',
                     'T2FSPOSTAX',
                     'T1POSTAX',
                     'FLAIRPOSTAX',
                     'SWIAX',
                     'SWIMINIPAX'
            , 'ADCAX',
                     'T1MPRAGEMPRCOR',
                     'T1MPRAGEMPRAX'
                     # Add more sequences and criteria here
                     },
        'LENGTH': {12, 13}
    },

    'B3A': {
        'PRESENCE': {'DWIAX',
                     'FLAIRSAG',
                     'T1MPRAGESAG',
                     'T2*AX',
                     'T1AX',
                     'T2FSPOSTAX',
                     'T1POSTAX',
                     'T1MPRAGEPOSTSAG',
                     'SWIAX',
                     'SWIMINIPAX'
            , 'ADCAX',
                     'T1MPRAGEMPRCOR',
                     'T1MPRAGEMPRAX',
                     'FLAIRMPRAX',
                     'FLAIRMPRCOR'

                     # Add more sequences and criteria here
                     },
        'LENGTH': {13, 14}
    },
    'DWI2DAX': {
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











# I am going to use this

# Assuming you have a dataframe named 'df' with a cdolumn named 'sequence'
groups = df.groupby("StudyInstanceUID")
# For each group, perform compliance checks for each sequence
for group_name, group_data in groups:
    print("Processing group:", group_name)
    print(group_data[['StudyDescription', 'SeriesDescription', 'sequence']])
    # Ask the user if they want to go with a default protocol compliance check or custom protocol
    protocol_option = input(
        "Do you want to go with a default protocol compliance check or custom protocol? Enter 'default' or 'custom': ")
    score = 10
    # If the user chooses default protocol, ask for the protocol name and search it in the criteria dictionary
    if protocol_option == "default":
        protocol_name = input("Enter the protocol name: ")
        df_length = len(group_data)
        sequence_names = group_data['sequence'].tolist()
        unmatched_sequence_names = []

        # Initialize an empty DataFrame to store results
        results_data = []

        if protocol_name in criteria.keys():
            presence = criteria[protocol_name]['PRESENCE']
            for sequence_name in sequence_names:
                if sequence_name not in presence:
                    unmatched_sequence_names.append(sequence_name)
            print("Unmatched sequence names:", unmatched_sequence_names)
            length_range = criteria[protocol_name]['LENGTH']
            if df_length in length_range:
                print("Dataframe length is within range:", df_length)
            else:
                print("Dataframe length is out of range:", df_length, "instead of", length_range)

            for index, row in group_data.iterrows():
                sequence_name = row["sequence"]
                orientation = row["MRAcquisitionType"]
                fov = row["FOV"]
                pixel_area = row["pixel_area"]
                thickness = row["SliceThickness"]
                gap = row["SliceGap"]
                coverage = row["vol_cov"]

                if sequence_name in criteria.keys():
                    orientation_result = f"correct" if orientation == criteria[sequence_name][
                        "Orientation"] else f"incorrect (expected: {criteria[sequence_name]['Orientation']})"

                    fov_result = f"within range" if criteria[sequence_name]["FOV"][0] <= fov <= \
                                                    criteria[sequence_name]["FOV"][
                                                        1] else f"out of range (expected: {criteria[sequence_name]['FOV']})"
                    pixel_area_result = "less than or equal to max pixel area" if pixel_area <= criteria[sequence_name][
                        "PixelArea"] else f"greater than max pixel area (expected: {criteria[sequence_name]['PixelArea']})"
                    thickness_result = "less than or equal to max thickness" if thickness <= criteria[sequence_name][
                        "Thickness"] else f"greater than max thickness (expected: {criteria[sequence_name]['Thickness']})"
                    gap_result = "less than or equal to max gap" if gap <= criteria[sequence_name][
                        "Gap"] else f"greater than max gap (expected: {criteria[sequence_name]['Gap']})"
                    coverage_result = "greater than or equal to min coverage" if coverage >= criteria[sequence_name][
                        "Coverage"] else f"less than min coverage (expected: {criteria[sequence_name]['Coverage']})"

                    if (orientation_result.startswith("incorrect")).any() or \
                            (fov_result.startswith("out of range")).any() or \
                            (pixel_area_result.startswith("greater than max pixel area")).any() or \
                            (not pd.isna(kk['SliceThickness']) and thickness_result.startswith(
                                "greater than max thickness")).any() or \
                            (not pd.isna(kk['SpacingBetweenSlices']) and not pd.isna(
                                kk['SliceThickness']) and gap_result.startswith("greater than max gap")).any() or \
                            (not pd.isna(kk['SpacingBetweenSlices']) and not pd.isna(
                                kk['SliceThickness']) and coverage_result.startswith("less than min coverage")).any():


                            score = score - 1

                    results_data.append(
                        [sequence_name, orientation_result, fov_result, pixel_area_result, thickness_result, gap_result,
                         coverage_result])

            # Create a DataFrame from the results_data list
            results_df = pd.DataFrame(results_data,
                                      columns=["Sequence Name", "Orientation", "FOV", "Pixel Area", "Thickness", "Gap",
                                               "Coverage"])

            # Print the DataFrame
            print(results_df)

        else:
            print(sequence_name, "is not in the criteria dictionary")
        # Count the number of series in the first group and check it with a given number

    # If the user chooses custom protocol, ask for their own criteria and check them accordingly
    if protocol_option == "custom":
        # Ask for their own criteria and store them in a dictionary
        print("write sequence name as shown in sequence column")


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

            return custom_criteria


        # Call the function to create a custom criteria dictionary
        custom_criteria = create_custom_criteria()

        # Print the custom criteria dictionary
        print(custom_criteria)
        df_length = len(group_data)
        sequence_names = group_data['sequence'].tolist()
        unmatched_sequence_names = []

        # Initialize an empty DataFrame to store results
        results_data = []

        if protocol_name in custom_criteria.keys():
            presence = custom_criteria[protocol_name]['PRESENCE']
            for sequence_name in sequence_names:
                if sequence_name not in presence:
                    unmatched_sequence_names.append(sequence_name)
            print("Unmatched sequence names:", unmatched_sequence_names)
            length_range = custom_criteria[protocol_name]['LENGTH']
            if df_length in length_range:
                print("Dataframe length is within range:", df_length)
            else:
                print("Dataframe length is out of range:", df_length, "instead of", length_range)

            for index, row in group_data.iterrows():
                sequence_name = row["sequence"]
                orientation = row["MRAcquisitionType"]
                fov = row["FOVy"]
                pixel_area = row["pixel_area"]
                thickness = row["SliceThickness"]
                gap = row["SliceGap"]
                coverage = row["vol_cov"]

                if sequence_name in custom_criteria.keys():
                    orientation_result = f"correct" if orientation == custom_criteria[sequence_name][
                        "Orientation"] else f"incorrect (expected: {custom_criteria[sequence_name]['Orientation']})"
                    fov_result = f"within range" if custom_criteria[sequence_name]["FOV"][0] <= fov <= \
                                                    custom_criteria[sequence_name]["FOV"][
                                                        1] else f"out of range (expected: {custom_criteria[sequence_name]['FOV']})"
                    pixel_area_result = "less than or equal to max pixel area" if pixel_area <= \
                                                                                  custom_criteria[sequence_name][
                                                                                      "PixelArea"] else f"greater than max pixel area (expected: {custom_criteria[sequence_name]['PixelArea']})"
                    thickness_result = "less than or equal to max thickness" if thickness <= \
                                                                                custom_criteria[sequence_name][
                                                                                    "Thickness"] else f"greater than max thickness (expected: {custom_criteria[sequence_name]['Thickness']})"
                    gap_result = "less than or equal to max gap" if gap <= custom_criteria[sequence_name][
                        "Gap"] else f"greater than max gap (expected: {custom_criteria[sequence_name]['Gap']})"
                    coverage_result = "greater than or equal to min coverage" if coverage >= \
                                                                                 custom_criteria[sequence_name][
                                                                                     "Coverage"] else f"less than min coverage (expected: {custom_criteria[sequence_name]['Coverage']})"
                    if orientation_result.startswith("incorrect") or fov_result.startswith(
                            "out of range") or pixel_area_result.startswith("greater than max pixel area") or (
                            not pd.isna(kk['SliceThickness']) and thickness_result.startswith(
                            "greater than max thickness")) or (not pd.isna(kk['SpacingBetweenSlices']) and not pd.isna(
                            kk['SliceThickness']) and gap_result.startswith("greater than max gap")) or (
                            not pd.isna(kk['SpacingBetweenSlices']) and not pd.isna(
                            kk['SliceThickness'] and coverage_result.startswith("less than min coverage"))):
                        score = score - 1
                    results_data.append(
                        [sequence_name, orientation_result, fov_result, pixel_area_result, thickness_result, gap_result,
                         coverage_result])

            # Create a DataFrame from the results_data list
            results_df = pd.DataFrame(results_data,
                                      columns=["Sequence Name", "Orientation", "FOV", "Pixel Area", "Thickness", "Gap",
                                               "Coverage"])

            # Print the DataFrame
            print(results_df)

        else:
            print(sequence_name, "is not in the criteria dictionary")
        # Count the number of series in the first group and check it with a given number


    # If the user enters an invalid option, show an error message and exit
    else:
        print("Invalid option. Please enter 'default' or 'custom'.")
    print(score, "is the score")


