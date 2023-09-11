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

-**Inputs**
1. The program expects you to give the folder path of the parent folder where studies are stored. There is no different file that should be present other than Dicom.
2. After that, you must give input on whether you need to change the extension of files to .dcm or not.
3. If you select custom for protocol compliance, you need to input the sequence name like
   - weighting + _FS ( if fat suppressed) + _POST( if Contrast Bolus Agent used) + _MPR( if applied) + _MIP(if MIP used ) + _2D/_3D + _AX/_SAG/_COR(depending upon anatomical plane).
   - Name keeping in mind the sequence column of table output.
   

