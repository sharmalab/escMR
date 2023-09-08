import pandas as pd
import os
import tqdm
import numpy as np
import ast

df = pd.read_csv('/labs/colab/himanshu_GSoC/data/Final_df.csv')

df = df.drop_duplicates(subset='SeriesInstanceUID')

empty_rows = df[df['SeriesDescription'].isna() | df['SeriesDescription'].isnull()]
df = df.drop(empty_rows.index)



df['SeriesDescription'] = df['SeriesDescription'].str.upper()
df['StudyDescription'] = df['StudyDescription'].str.upper()
df['ProtocolName'] = df['ProtocolName'].str.upper()

# List of study descriptions to exclude
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
        return 'Sagittal'
    elif plane[1] == 1:
        return 'Coronal'
    elif plane[2] == 1:
        return 'Axial'
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


df = add_anatomical_plane_column(df)
k =df[df['Anatomical Plane'] == 'Unknown']
df = df.drop(k.index)
df.reset_index(drop=True, inplace=True)

