import pandas as pd
import os
import tqdm
import pandas as pd
import ast
import numpy as np
from tqdm import tqdm
import time



from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import warnings

# Suppress PerformanceWarning
warnings.filterwarnings("ignore", category=pd.PerformanceWarning)
pd.options.display.max_columns = None
folder_path = input("Enter the folder path: ")
df = pd.read_csv(folder_path)

df = df.drop_duplicates(subset='SeriesInstanceUID')

empty_rows = df[df['SeriesDescription'].isna() | df['SeriesDescription'].isnull()]
df = df.drop(empty_rows.index)

df['SeriesDescription'] = df['SeriesDescription'].str.upper()
df = df[df['SeriesDescription'] != 'DYNASUITE RESULTS']
df['StudyDescription'] = df['StudyDescription'].str.upper()
df['ProtocolName'] = df['ProtocolName'].str.upper()

# List of study descriptions to exclude
#remove the study description from below list if you want to train on that study also
excluded_study_descriptions = [

    "MRA NECK W  WO CONTRAST",
    "MRA NECK WO CONTRAST",
    "MRI ABDOMEN W  WO CONTRAST",
    "MRI ABDOMINAL OUTSIDE IMAGE REF ONLY",
    "MRI FACE W WO CONT",
    "MRI HAND WO CONTRAST RIGHT",

    "MRI MSK OUTSIDE IMAGE REF ONLY",
    "MRI BRAIN FUNCTIONAL MAPPING",
    "MRI NEURO BRAIN OUTSIDE IMAGE CONSULT",
    "MRI NEURO BRAIN OUTSIDE IMAGE REF ONLY",

    "MRI NEURO OTHER OUTSIDE IMAGE CONSULT",
    "MRI NEURO OTHER OUTSIDE IMAGE REF ONLY",
    "MRI NEURO SPINE OUTSIDE IMAGE REF ONLY",
    "MRI MSK OUTSIDE IMAGE REF ONLY",
    "MRI NEURO OTHER OUTSIDE IMAGE CONSULT",
    "MRI NEURO OTHER OUTSIDE IMAGE REF ONLY",
    "MRI NEURO SPINE OUTSIDE IMAGE REF ONLY",

    "MRI PELVIS (GI/GU) W/ + W/O CONTRAST",

    "MRI SPINE CERVICAL W WO CONTRAST",
    "MRI SPINE CERVICAL WO CONTRAST",
    "MRI SPINE LUMBAR W WO CONTRAST",
    "MRI SPINE LUMBAR WO CONTRAST",
    "MRI SPINE THORACIC W WO CONTRAST",
    "MRI SPINE THORACIC WO CONTRAST",
    "MRI WRIST WO CONTRAST LEFT",
    "MRI WRIST WO CONTRAST RIGHT",
    "UNSPECIFIED MERGE ONLY MR"
    "UNSPECIFIED MERGE ONLY MR",

    "UNSPECIFIED MR BRAIN",
    "UNSPECIFIED MERGE ONLY MR"

]

# Filter the DataFrame using .loc
df = df.loc[~df["StudyDescription"].isin(excluded_study_descriptions)]

# Assuming 'df' is the original DataFrame containing the 'StudyDescription' column
# Reset the index to avoid any potential issues with duplicate indices
df.reset_index(drop=True, inplace=True)

import numpy as np

import math

def file_plane(IOP):
    if isinstance(IOP, str):
        IOP_values = [float(x.strip().strip('"')) if x != 'NaN' else math.nan for x in IOP.strip('()').split(',')]
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

selected_column = df["SeriesDescription"]
tokenized_column = selected_column.str.split(r'[\s_()\[\]]+')

df['tokenized column'] = tokenized_column
# Display the tokenized column

# Flatten the series of lists into a single list
tokenized_column_flat = tokenized_column.explode()

# Get unique tokens
unique_tokens = tokenized_column_flat.unique()

# CODE USE TO LABELLING


# If a token of series description adds meaning or can help in differentiating from other sequences then add it here.
# For further processing it will work on below tokens only
tokens_to_search = ['T1', 'T2', 'SWI', 'GRE', 'VIBE', 'CISS', 'GRE3D', 'FLAIR', 'MPRAGE', 'DIR', 'DWI', 'ADC', 'TRACEW',
                    'MIP', 'MPR', 'LOCALIZER', 'AAHSCOUT', 'TOF', 'MRA', 'COW', 'ANTERIOR', 'ANT', 'ENT', 'POSTERIOR',
                    'ENTIRE''CAROTID', 'BRAVO', 'T2*', 'PROPELLER', 'SCOUT', 'DIFF', 'DIFFUSION', 'DTI', '3PL',
                    'LOC3D ', 'T2FS', 'TWIST', 'GRE3D', 'SPACE', 'PHASE', 'PHA', 'PERFUSION', 'PERF', '3-PLANE', 'LT',
                    'RT', 'MS-DIR', 'CISS', 'VIBE', 'BLADE', 'SUB', 'TWIST', 'TRICKS', 'CAROTID', 'TOF3D', 'TOF2D',
                    'COROTID', 'CAROTIDS', 'CAROTID', 'MRV', 'T1FS', 'FIESTA', 'FIESTA-C', 'FGATIR']

# Initialize new columns with empty values
df['Matching Tokens'] = ''
df['Calculation Result'] = ''
# Convert 'Matching Tokens' column to list type if it was initially empty
df['Matching Tokens'] = df['Matching Tokens'].apply(lambda x: [] if pd.isnull(x) else x)
# Initialize variables to track the longest iteration
max_elapsed_time = 0
max_elapsed_idx = 10
for idx in tqdm(range(len(df)), total=len(df), desc="Processing Rows"):
    # Print debug information about the current row being processed
    print(f"Processing row {idx + 1}/{len(df)}")

    # Check if the current index is within the range [2250, 3350]

    # Retrieve matching tokens
    matching_tokens = [token for token in df.loc[idx, 'tokenized column'] if token in tokens_to_search]

    # Perform calculations and store the result in the 'Calculation Result' column
    calculation_result = None

    start_time = time.time()

    # ARDL & MEDIUM & RESOLVE do not add any information so is not added here
    #Add rules what do you want to do with that token whether change it, remove it , add with something else
    if 'T1FS' in matching_tokens:
        # Delete 'T1FS' token from the 'Matching Tokens' list
        matching_tokens.remove('T1FS')
        matching_tokens.append('T1')
    if 'T1' in matching_tokens and 'MPRAGE' in matching_tokens:
        # Delete 'T1' and 'MPRAGE' tokens from the 'Matching Tokens' list
        matching_tokens.remove('T1')
        matching_tokens.remove('MPRAGE')
        matching_tokens.append('T1_MPRAGE')
    if 'MPRAGE' in matching_tokens:
        # Delete 'MPRAGE' token from the 'Matching Tokens' column and replace with 'T1_MPRAGE'
        matching_tokens.remove('MPRAGE')
        matching_tokens.append('T1_MPRAGE')

    if 'T2FS' in matching_tokens:
        # Delete 'T2FS' token from the 'Matching Tokens' column and replace with 'T2'
        matching_tokens.remove('T2FS')
        matching_tokens.append('T2')

    if 'GRE3D' in matching_tokens:
        # Delete 'GRE3D' token from the 'Matching Tokens' column and replace with 'GRE'
        matching_tokens.remove('GRE3D')
        matching_tokens.append('GRE')

    if 'LOCALIZER' in matching_tokens or 'SCOUT' in matching_tokens or 'LOC' in matching_tokens or '3PL' in matching_tokens or 'LOC3D' in matching_tokens or '3-PLANE' in matching_tokens:
        # Delete relevant tokens from the 'Matching Tokens' column and replace with 'SCOUT'
        for token in ['LOCALIZER', 'LOC', '3PL', 'SCOUT', '3-PLANE', 'AAHSCOUT']:
            if token in matching_tokens:
                matching_tokens.remove(token)
        matching_tokens.append('SCOUT')

    # AS BLADE IS NOT IN THE PROTOCOL TEMPLATE SO WE CAN REMOVE IT AND PROPELLER ALSO AS BOTH ARE SAME
    if 'PROPELLER' in matching_tokens:
        # Delete 'PROPELLER' token from the 'Matching Tokens' column and replace with 'BLADE'
        matching_tokens.remove('PROPELLER')

    if 'BLADE' in matching_tokens:
        # Delete 'PROPELLER' token from the 'Matching Tokens' column and replace with 'BLADE'
        matching_tokens.remove('BLADE')

    if 'FIESTA-C' in matching_tokens:
        # Delete 'FIESTA-C' token from the 'Matching Tokens' column and replace with 'FIESTA'
        matching_tokens.remove('FIESTA-C')
        matching_tokens.append('FIESTA')

    if 'T2' in matching_tokens and 'FLAIR' in matching_tokens:
        # Delete 'T2' token from the 'Matching Tokens' column
        matching_tokens.remove('T2')

    if 'T2' in matching_tokens and 'GRE' in matching_tokens:
        # Delete 'T2' and 'GRE' tokens from the 'Matching Tokens' column and replace with 'T2*'
        matching_tokens.remove('T2')
        matching_tokens.remove('GRE')
        matching_tokens.append('T2*')
    if 'T2*' in matching_tokens and 'GRE' in matching_tokens:
        # Delete 'T2' token from the 'Matching Tokens' column
        matching_tokens.remove('GRE')
    # aas there is no need of GRE we can remoce it
    if 'GRE' in matching_tokens:
        matching_tokens.remove('GRE')
    if 'T1' in matching_tokens and 'VIBE' in matching_tokens:
        # Delete 'T2' token from the 'Matching Tokens' column
        matching_tokens.remove('T1')
    if 'T2' in matching_tokens and 'CISS' in matching_tokens:
        # Delete 'T2' token from the 'Matching Tokens' column
        matching_tokens.remove('T2')
    if 'DIFF' in matching_tokens or 'DIFFUSION' in matching_tokens:
        # Delete 'DIFF' and 'DIFFUSION' tokens from the 'Matching Tokens' column and replace with 'DWI'
        matching_tokens.remove('DIFF')
        matching_tokens.remove('DIFFUSION')
        matching_tokens.append('DWI')

    if 'DWI' in matching_tokens and 'ADC' in matching_tokens:
        # Delete 'DWI' token from the 'Matching Tokens' column
        matching_tokens.remove('DWI')

    if 'MS-DIR' in matching_tokens:
        # Delete 'MS-DIR' token from the 'Matching Tokens' column and replace with 'DIR'
        matching_tokens.remove('MS-DIR')
        matching_tokens.append('DIR_SPACE')
    if 'DIR' in matching_tokens:
        matching_tokens.remove('DIR')
        matching_tokens.append('DIR_SPACE')
    if 'DIR_SPACE' in matching_tokens and 'SPACE' in matching_tokens:
        matching_tokens.remove('SPACE')
    if 'CAROTIDS' in matching_tokens or 'COROTID' in matching_tokens or 'CAROTID' in matching_tokens:
        # Delete relevant tokens from the 'Matching Tokens' column and replace with 'TOF'
        for token in ['CAROTID', 'COROTID', 'CAROTIDS']:
            if token in matching_tokens:
                matching_tokens.remove(token)
        matching_tokens.append('TOF')

    if 'TOF2D' in matching_tokens:
        # Delete 'TOF2D' and 'TOF3D' tokens from the 'Matching Tokens' column and replace with 'TOF'
        matching_tokens.remove('TOF2D')

        matching_tokens.append('TOF')
    if 'TOF3D' in matching_tokens:
        # Delete 'TOF2D' and 'TOF3D' tokens from the 'Matching Tokens' column and replace with 'TOF'

        matching_tokens.remove('TOF3D')
        matching_tokens.append('TOF')
    if 'MRA' in matching_tokens and 'TOF' in matching_tokens:
        # Delete 'MRA' token from the 'Matching Tokens' column
        matching_tokens.remove('MRA')

    if 'MRA' in matching_tokens or 'ENTIRE' in matching_tokens:
        # Delete relevant tokens from the 'Matching Tokens' column and replace with 'COW_ENT'
        for token in ['ENTIRE', 'ENT', 'COW']:
            if token in matching_tokens:
                matching_tokens.remove(token)
        matching_tokens.append('COW_ENT')

    # AS TWIST IS TIME RESOLVE 3D MRA
    if 'TWIST' in matching_tokens or 'TRICKS' in matching_tokens:
        # Delete relevant tokens from the 'Matching Tokens' column and replace with 'MRA'
        for token in ['TWIST', 'TRICKS']:
            if token in matching_tokens:
                matching_tokens.remove(token)
        matching_tokens.append('MRA')

    if 'TWIST' in matching_tokens and 'VIBE' in matching_tokens:
        # Delete 'TWIST' token from the 'Matching Tokens' column
        matching_tokens.remove('TWIST')

    if 'ENT' in matching_tokens or 'ENTIRE' in matching_tokens:
        # Delete relevant tokens from the 'Matching Tokens' column and replace with 'COW_ENT'
        for token in ['ENTIRE', 'ENT', 'COW']:
            if token in matching_tokens:
                matching_tokens.remove(token)
        matching_tokens.append('COW_ENT')

    if 'POSTERIOR' in matching_tokens:
        # Delete 'POSTERIOR' token from the 'Matching Tokens' column and replace with 'COW_POSTERIOR'
        matching_tokens.remove('POSTERIOR')
        matching_tokens.append('TOF_POSTERIOR')

    if 'ANTERIOR' in matching_tokens and ('COW' in matching_tokens or 'ANT' in matching_tokens):
        # Delete relevant tokens from the 'Matching Tokens' column and replace with 'COW_ANT'
        for token in ['ANTERIOR', 'COW', 'ANT']:
            if token in matching_tokens:
                matching_tokens.remove(token)
        matching_tokens.append('COW_ANT')
    if 'COW' in matching_tokens:
        # Delete 'T2' token from the 'Matching Tokens' column
        matching_tokens.remove('COW')
        matching_tokens.append('TOF')
    if 'T1' in matching_tokens and 'MPR' in matching_tokens and 'GRE' in matching_tokens:
        # Delete 'T2' token from the 'Matching Tokens' column
        matching_tokens.remove('GRE')
    if 'T1' in matching_tokens and 'SCOUT' in matching_tokens and 'SCOUT' in matching_tokens:
        # Delete 'T2' token from the 'Matching Tokens' column
        matching_tokens.remove('T1')
    if 'MRA' in matching_tokens and 'COW_ENT' in matching_tokens:
        # Delete 'T2' token from the 'Matching Tokens' column
        matching_tokens.remove('MRA')
    if 'ANTERIOR' in matching_tokens:
        matching_tokens.remove('ANTERIOR')
        matching_tokens.append('ANT')
    if 'ANT' in matching_tokens:
        # Delete 'T2' token from the 'Matching Tokens' column
        matching_tokens.append('COW')
    if 'FLAIR' in matching_tokens and 'SPACE' in matching_tokens:
        # Delete 'T2' token from the 'Matching Tokens' column
        matching_tokens.remove('FLAIR')
        matching_tokens.remove('SPACE')
        matching_tokens.append('T2_SPACE')
    if 'DWI' in matching_tokens and 'TRACEW' in matching_tokens:
        # Delete 'T2' token from the 'Matching Tokens' column
        matching_tokens.remove('TRACEW')
    if 'PERFUSION' in matching_tokens:
        # Delete 'T2' token from the 'Matching Tokens' column
        matching_tokens.remove('PERFUSION')
        matching_tokens.append('PERF')
    if 'COW_ENT' in matching_tokens:
        # Delete 'T2' token from the 'Matching Tokens' column
        matching_tokens.remove('COW_ENT')
        matching_tokens.append('TOF_ENT')
    if 'COW_ANT' in matching_tokens:
        # Delete 'T2' token from the 'Matching Tokens' column
        matching_tokens.remove('COW_ANT')
        matching_tokens.append('TOF_ANT')
    if 'COW' in matching_tokens:
        # Delete 'T2' token from the 'Matching Tokens' column
        matching_tokens.remove('COW')
        matching_tokens.append('TOF')

    elapsed_time = time.time() - start_time
    if elapsed_time > max_elapsed_time:
        max_elapsed_time = elapsed_time
        max_elapsed_idx = idx

    # Check if the current iteration took more than 60 seconds

    # Merge all tokens left in 'matching_tokens' into the 'Calculation Result'
    if not matching_tokens:
        df.at[idx, 'Calculation Result'] = np.nan

    df.at[idx, 'Calculation Result'] = '_'.join(matching_tokens)

    # Add a line break for readability
    print()

# Print the updated DataFrame
print(df)

# List of strings to be deleted
# here if  you want to remove some rows which are not helpful for training or you don't want to work with it
strings_to_delete = ['', 'MPR', 'MIP_TOF_TOF', 'MIP_SCOUT', 'FLAIR_MPR_FLAIR', 'T1_TOF', 'AAHSCOUT', 'MIP_TOF_TOF',
                     'MIP_SCOUT', 'FLAIR_MPR_FLAIR', 'T1_TOF', 'AAHSCOUT', 'VIBE_MIP_MRA', 'VIBE_MRA', 'TRACEW',
                     'TOF_ANT_TOF', 'TOF_SCOUT', 'MPR_TOF_ENT', 'RT_MRV', 'MRV_MPR', 'SUB_MIP_MRA_TOF_ENT',
                     'MIP_SUB_TOF_ENT', 'MIP_TOF_ENT', 'FA', 'AAHSCOUT_MPR ', 'MIP ', 'GRE ', 'T1_MIP_TOF',
                     'MIP_T1_MPRAGE', 'MPR_TOF', 'TOF_TOF', 'FLAIR_MPR_MPR', 'BRAVO', 'MPR_FLAIR']

# Delete rows containing strings from the list
df = df[~df['Calculation Result'].isin(strings_to_delete)]
# If the output is not coming as expected then you can change the name of sequence here like below examples if tokens are not in order
replacement_rules = {
    'MIP_TOF': 'TOF_MIP',
    'TOF_MIP_TOF': 'TOF_MIP',
    'ANT_TOF_TOF': 'ANT_TOF',
    'T2_SPACE_MPR': 'MPR_T2_SPACE',
    'TOF_POSTERIOR_TOF': 'POSTERIOR_TOF',
    'TOF_ANT': 'ANT_TOF',
    'LT': 'LT_TOF',
    'RT': 'RT_TOF',
    'DTI_TRACEW': 'DTI',
    'DTI_ADC': 'DTI',
    'T2_FIESTA': 'FIESTA'
}

# Apply the replacement rules using .replace() with regex=True
df['Calculation Result'] = df['Calculation Result'].replace(replacement_rules, regex=True)
df.reset_index(drop=True, inplace=True)
#run it twice
replacement_rules = {
    'ADC': 'DWI',
    'DWI_T2' : 'DWI',
    'ADC_T2' : 'DWI',
    'T1_MPR' : 'T1' ,
    'CISS_MPR' : 'CISS',
    'MRV_SUB' : 'MRV',
    'SUB_TOF' : 'TOF',
    'MRV_MIP' : 'MRV',
    'SUB_MRA' : 'MRA',
    'SUB_MIP_MRA' : 'MRA',
       'MRV_SUB_MIP' : 'MRV',
    'MPR_T1_MPRAGE': 'T1_MPRAGE',
    'MPR_SCOUT': 'SCOUT',
    'FLAIR_MPR' : 'FLAIR',
    'MPR_T2_SPACE': 'T2_SPACE',
    'POSTERIOR_TOF': 'TOF',
    'TOF_MIP' : 'TOF',
    'ANT_TOF': 'TOF',
    'TOF_ENT' : 'TOF',
    'SUB_TOF_ENT' : 'TOF',
     'SUB_TOF_MIP_ENT' : 'TOF',
    'RT_TOF' : 'TOF',
    'LT_TOF' : 'TOF',
    'TOF_MIP' : 'TOF',
    'TOF_POSTERIOR': 'TOF',
    'AAHSCOUT_MPR' : 'SCOUT',
       'TOF_TOF' : 'TOF',
    'RT_TOF_TOF':'TOF',
    'LT_TOF_TOF': 'TOF',
    'FIESTA_MPR': 'FIESTA',
    'MPR_FIESTA' : 'FIESTA',
    'MPR_DIR_SPACE': 'DIR_SPACE',
    'T2_FIESTA': 'FIESTA',
   'MPR_MPR_T1AGE' : 'T1_MPRAGE',
    'MPR_T1AGE' : 'T1_MPRAGE',
    'T1AGE' : 'T1_MPRAGE'
}

# Apply the replacement rules using .replace() with regex=True
df['Calculation Result'] = df['Calculation Result'].replace(replacement_rules, regex=True)
# here we are filling empty rows, so we can transform them later
df['ScanOptions'].fillna('dukh', inplace=True)
df['SequenceVariant'].fillna('dard', inplace=True)
df['ScanningSequence'].fillna('peeda', inplace=True)
df['ImageType'].fillna('kasht', inplace=True)
df['ScanOptions'].fillna('avsaad',inplace = True)

# working for large data
unique_values = df['ImageType'].unique()

# Create binary features based on unique values
for value in unique_values:
    value_str = value.strip("()").replace("'", "").replace(" ", "")
    features = value_str.split(',')
    for feature in features:
        feature = feature.strip()
        df[feature] = df['ImageType'].apply(lambda x: int(feature in x))


# working for large data
unique_values = df['ScanOptions'].unique()

# Create binary features based on unique values
for value in unique_values:
    value_str = value.strip("()").replace("'", "").replace(" ", "")
    features = value_str.split(',')
    for feature in features:
        feature = feature.strip()
        df[feature] = df['ScanOptions'].apply(lambda x: int(feature in x))

# Convert 'ImageType' column to uppercase
df['SequenceVariant'] = df['SequenceVariant'].str.upper()

# Tokenize the 'ImageType' column
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


# Convert 'ImageType' column to uppercase
df['ScanningSequence'] = df['ScanningSequence'].str.upper()

# Tokenize the 'ImageType' column
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


df['ContrastBolusAgent'] = df['ContrastBolusAgent'].apply(lambda x: 1 if isinstance(x, str) and x.strip() else 0)

# If you want to replace NaN with 0 before applying the above operation, use 'fillna':
df['ContrastBolusAgent'] = df['ContrastBolusAgent'].fillna(0).apply(lambda x: 1 if x == 1 else 0)
def extract_pixel_spacing(string):
    """Extract the numeric part of a string representing pixel spacing."""
    match = re.search(r'\d+\.*\d*', str(string))
    if match:
        return float(match.group())
    else:
        return np.nan


df['PixelSpacing'] = df['PixelSpacing'].apply(extract_pixel_spacing)
df = df.dropna(subset=['PixelSpacing'])
one_hot_encoded = pd.get_dummies(df['MagneticFieldStrength'])

# Concatenate the one-hot encoded DataFrame with the original DataFrame
df = pd.concat([df, one_hot_encoded], axis=1)

one_hot_encoded = pd.get_dummies(df['PhotometricInterpretation'])

# Concatenate the one-hot encoded DataFrame with the original DataFrame
df = pd.concat([df, one_hot_encoded], axis=1)

one_hot_encoded = pd.get_dummies(df['MRAcquisitionType'])

# Concatenate the one-hot encoded DataFrame with the original DataFrame
df = pd.concat([df, one_hot_encoded], axis=1)
df['InversionTime'] = df['InversionTime'].fillna(0)
df.fillna(100000, inplace=True)
import pandas as pd

# Your list of strings
target_strings = ['DWI',  'T2', 'FLAIR', 'T2*', 'T1',  'SCOUT',
       'VIBE', 'CISS', 'TOF',  'DIR_SPACE', 'T2_SPACE', 'PERF',
       'DTI', 'FGATIR', 'T1_FLAIR', 'MRV', 'FIESTA', 'T1_MPRAGE',
       'MRA']

# Assuming df is your DataFrame
filtered_rows = df[df.apply(lambda row: any(any(target in str(cell) for target in target_strings)
                                          if isinstance(cell, str) else False
                                      for cell in row), axis=1)]
# filtered_rows now contains the rows that match your criteria
print(['Calculation Result'].unique())
filtered_rows.reset_index(drop=True, inplace=True)

import pandas as pd

y_train = filtered_rows['Calculation Result']
filtered_rows.columns = filtered_rows.columns.astype(str)

# Convert target variable to strings
y = y_train.astype(str)
x_train = filtered_rows[[
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

# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Fit and transform the target variable 'y' to numerical values
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(x_train, y_encoded, test_size=0.2, random_state=42)

# Create the RandomForestClassifier
rf_model = RandomForestClassifier()

# Define the hyperparameter grid for tuning
param_grid = {
    'n_estimators': [300],
    'max_depth': [50],

    'criterion': ['entropy']
    # Add more hyperparameters you want to tune
}

# Create the GridSearchCV object with cross-validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1)

# Perform the hyperparameter tuning
grid_search.fit(X_train, y_train_encoded)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Get the best model with the best hyperparameters
best_model = grid_search.best_estimator_

# Now you can use the best_model for predictions and evaluation
y_pred_encoded = best_model.predict(X_test)

# Inverse transform the numerical predictions back to their original string labels
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Now 'y_pred' contains the human-readable class labels as strings
print(y_pred)
gg =  best_model.predict(X_train)
predicted_labels = best_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test_encoded, predicted_labels)
print("Accuracy:", accuracy)


accuracy = accuracy_score(y_test_encoded, predicted_labels)
conf_matrix = confusion_matrix(y_test_encoded,predicted_labels)
class_report = classification_report(y_test_encoded,predicted_labels)  # Automatically handles multiclass



# Get ROC curve data (ROC curve is not directly applicable to multiclass problems)
# You may need to compute ROC curves for each class separately

# Print the metrics (same as before)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

import joblib
joblib.dump(best_model, 'RandomForestModel.pkl')
