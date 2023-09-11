# escMR: Exam Series Categorization for MR Images.


# Project Name: A Framework And Methodology To Categorize Exam Series.
## Description

This project was developed as part of the [Google Summer of Code (GSoC) 2023](https://summerofcode.withgoogle.com/programs/2023/projects/oC59dZpT) program. The primary objective of this project is to create a robust framework and methodology for categorizing MRI exam series of the brain based on DICOM metadata. Additionally, the framework aims to perform compliance checks on MRI Brain Studies.
Supported sequence weighting:
- 'DWI'
- 'T1_MPRAGE'
- 'T2'
- 'FLAIR'
- 'T2*'
- 'T1'
- 'SCOUT'
- 'VIBE'
- 'CISS'
- 'TOF'
- 'DIR_SPACE'
- 'T2_SPACE'
- 'PERF'
- 'DTI'
- 'FGATIR'
- 'T1_FLAIR'
- 'MRV'
- 'FIESTA'
- 'MRA'

## UserGuide
**Installation**

1. After downloading the package, use the `cd` command to navigate to the directory where your Python script is located or the directory where your project files are located:

    ```bash
    cd path/to/your/project
    ```

2. Use the below command to install all requirements from a `Requirements.txt` file:

    ```bash
    pip install -r Requirements.txt
    ```

3. Now enter the following command to run your Python script:

    ```bash
    python Code.py
    ```
## Inputs

To use this program, you'll need to provide the following inputs:

1. **Folder Path**: The program expects you to specify the folder path of the parent directory where your MRI study data is stored. Ensure that the only files present in this folder are DICOM files.

2. **Change File Extension (Optional)**: You can choose whether you want to change the file extensions of the DICOM files to `.dcm`. This step is optional and depends on your preference.

3. **Custom Protocol Compliance (Optional)**: If you select the custom protocol compliance option, you will need to input the sequence name according to the following format:
   - **Sequence Name Format**: 
     - `weighting` + `_FS` (if fat suppressed) + `_POST` (if Contrast Bolus Agent used) + `_MPR` (if applied) + `_MIP` (if MIP used) + `_2D/_3D` + `_AX/_SAG/_COR` (depending on the anatomical plane).
   - Keep in mind that the sequence name should match the values in the sequence column of the table output.

These inputs allow you to customize how the program processes and categorizes your MRI study data. Make sure to provide accurate information to ensure the desired results.


