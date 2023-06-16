import pydicom
import numpy as np

def file_plane(IOP):
    IOP_round = [round(x) for x in IOP]
    plane = np.cross(IOP_round[0:3], IOP_round[3:6])
    plane = [abs(x) for x in plane]
    if plane[0] == 1:
        return "SAG"
    elif plane[1] == 1:
        return "COR"
    elif plane[2] == 1:
        return "AXI"

def add_anatomical_plane_column(df):
    # Create a copy of the DataFrame
    df_copy = df.copy()

    # Iterate over each row in the DataFrame
    anatomical_planes = []
    for index, row in df_copy.iterrows():
        IOP = row['Image Orientation (Patient)']
        anatomical_plane = file_plane(IOP)
        anatomical_planes.append(anatomical_plane)

    # Add the anatomical plane column to the copied DataFrame
    df_copy['Anatomical Plane'] = anatomical_planes

    return df_copy

# Assuming you have a DataFrame called 'df' containing DICOM metadata
df_with_plane = add_anatomical_plane_column(df)
print(df_with_plane)
