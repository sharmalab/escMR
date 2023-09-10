# escMR: Exam Series Categorization for MR Images.


# Project Name: A Framework And Methodology To Categorize Exam Series.

## Description

This project was developed as part of the [Google Summer of Code (GSoC) 2023](https://summerofcode.withgoogle.com/programs/2023/projects/oC59dZpT) program. The primary objective of this project is to create a robust framework and methodology for categorizing MRI exam series of the brain based on DICOM metadata. Additionally, the framework aims to perform compliance checks on MRI Brain Studies.
List of studies supported
- MRA HEAD W/ WO CONTRAST
- MRI BRAIN SEIZURE PROTOCOL W+ W/O CONT
- MRI BRAIN SEIZURE PROTOCOL WO CONT
- MRI BRAIN W + W/O CONTRAST (PHOTON)
- MRI BRAIN W CONTRAST
- MRI BRAIN W WO CONTRAST
- MRI BRAIN W/ + W/O CONTRAST (DEMENTIA)
- MRI BRAIN W/ + W/O CONTRAST (EPILEPSY)
- MRI BRAIN W/+ W/O CONTRAST (DBS SCREEN)
- MRI BRAIN W/+ W/O CONTRAST + PERFUSION
- MRI BRAIN W/O CONTRAST (DEMENTIA)
- MRI BRAIN W/O CONTRAST LIMITED
- MRI BRAIN WO CONTRAST
- MRI INT AUD CANAL W + W/O CONTRAST (IAC)
- MRI ORBITS W WO CONT
- MRI PITUITARY SELLA W WO CONTRAST
- MRI RAD PLANNING BRAIN
- MRI SPECTROSCOPY

  - Sequence weighting that can be identified:-'DWI', 'T1_MPRAGE', 'T2', 'FLAIR', 'T2*', 'T1', 'SCOUT', 'VIBE','CISS', 'TOF', 'DIR_SPACE', 'T2_SPACE', 'PERF', 'DTI', 'FGATIR','T1_FLAIR', 'MRV', 'FIESTA', 'MRA' .
## UserGuide
-**Inputs**
1. The program expects you to give the folder path of the parent folder where studies are stored. There is no different file that should be present other than Dicom.
2. After that, you must give input on whether you need to change the extension of files to .dcm or not.
3. If you select custom for protocol compliance, you need to input the sequence name like
   - weighting + FS ( if fat suppressed) + POST( if Contrast Bolus Agent used) + MPR( if applied) + MIP(if MIP used ) + 2D/3D + AX/SAG/COR(depending upon anatomical plane).
   - Name keeping in mind the sequence column of table output.
   
## Requirements
- Python 3.9
- NumPy version: 1.24.3
- Pandas version: 2.0.2
- scikit-learn (sklearn) version: 0.24.2
- joblib version: 1.3.2
