# CODE USE TO LABELLING

import pandas as pd
import ast
import numpy as np
from tqdm import tqdm
import time

#Add tokens which can help in distinguishing the sequences
tokens_to_search = ['T1', 'T2', 'SWI', 'GRE', 'VIBE', 'CISS', 'GRE3D', 'FLAIR', 'MPRAGE', 'DIR', 'DWI', 'ADC', 'TRACEW',
                    'MIP', 'MPR', 'LOCALIZER', 'AAHSCOUT', 'TOF', 'MRA', 'COW', 'ANT', 'ENT', 'POSTERIOR',
                    'ENTIRE''CAROTID', 'BRAVO', 'T2*', 'PROPELLER', 'SCOUT', 'DIFF', 'DIFFUSION', 'IAC', 'DTI', 'FA',
                    '3PL', 'LOC3D ', 'T2FS', 'TWIST', 'GRE3D', 'SPACE', 'PHASE', 'PHA', 'PERFUSION', 'PERF', '3-PLANE',
                    'LT', 'RT', 'MS-DIR', 'CISS', 'VIBE', 'BLADE', 'SUB', 'TWIST', 'TRICKS', 'CAROTID', 'TOF3D',
                    'TOF2D', 'COROTID', 'CAROTIDS', 'CAROTID', 'MRV', 'T1FS', 'FIESTA', 'FIESTA-C', 'FGATIR']

# Initialize new columns with empty values
df['Matching Tokens'] = ''
df['Calculation Result'] = ''
# Convert 'Matching Tokens' column to list type if it was initially empty
df['Matching Tokens'] = df['Matching Tokens'].apply(lambda x: [] if pd.isnull(x) else x)
# Initialize variables to track the longest iteration
max_elapsed_time = 0
max_elapsed_idx = 10
for idx in tqdm(np.arange(0, 10651), total=len(df), desc="Processing Rows"):
    # Print debug information about the current row being processed
    print(f"Processing row {idx + 1}/{len(df)}")



    # Retrieve matching tokens
    matching_tokens = [token for token in df.loc[idx, 'tokenized column'] if token in tokens_to_search]

    # Perform calculations and store the result in the 'Calculation Result' column
    calculation_result = None

    start_time = time.time()

    # ARDL & MEDIUM & RESOLVE do not add any information
    if 'T1FS' in matching_tokens:
        # Delete 'T1FS' token from the 'Matching Tokens' list
        matching_tokens.remove('T1FS')

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

    # AS BLADE IS NOT IN THE PROTOCOL TEMPLATE SO WE CAN REMOVE IT
    if 'PROPELLER' in matching_tokens:
        # Delete 'PROPELLER' token from the 'Matching Tokens' column and replace with 'BLADE'
        matching_tokens.remove('PROPELLER')
        matching_tokens.append('BLADE')

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

        matching_tokens.remove('COW')
        matching_tokens.append('TOF')
    if 'T1' in matching_tokens and 'MPR' in matching_tokens and 'GRE' in matching_tokens:

        matching_tokens.remove('GRE')
    if 'T1' in matching_tokens and 'SCOUT' in matching_tokens and 'SCOUT' in matching_tokens:

        matching_tokens.remove('T1')
    if 'MRA' in matching_tokens and 'COW_ENT' in matching_tokens:

        matching_tokens.remove('MRA')
    if 'ANT' in matching_tokens:

        matching_tokens.append('COW')
    if 'FLAIR' in matching_tokens and 'SPACE' in matching_tokens:

        matching_tokens.remove('FLAIR')
        matching_tokens.remove('SPACE')
        matching_tokens.append('T2_SPACE')
    if 'DWI' in matching_tokens and 'TRACEW' in matching_tokens:

        matching_tokens.remove('TRACEW')
    if 'PERFUSION' in matching_tokens:

        matching_tokens.remove('PERFUSION')
        matching_tokens.append('PERF')
    if 'COW_ENT' in matching_tokens:

        matching_tokens.remove('COW_ENT')
        matching_tokens.append('TOF_ENT')
    if 'COW' in matching_tokens:

        matching_tokens.remove('COW')
        matching_tokens.append('TOF')

    elapsed_time = time.time() - start_time
    if elapsed_time > max_elapsed_time:
        max_elapsed_time = elapsed_time
        max_elapsed_idx = idx


    # Merge all tokens left in 'matching_tokens' into the 'Calculation Result'
    if not matching_tokens:
        df.at[idx, 'Calculation Result'] = np.nan

    df.at[idx, 'Calculation Result'] = '_'.join(matching_tokens)

    # Add a line break for readability
    print()
print(df)


# List of strings to be deleted
strings_to_delete = ['','MPR','MIP_TOF_TOF', 'MIP_SCOUT', 'FLAIR_MPR_FLAIR', 'T1_TOF', 'AAHSCOUT', 'MIP_TOF_TOF', 'MIP_SCOUT', 'FLAIR_MPR_FLAIR', 'T1_TOF', 'AAHSCOUT', 'VIBE_MIP_MRA', 'VIBE_MRA', 'TRACEW', 'TOF_ANT_TOF', 'TOF_SCOUT', 'MPR_TOF_ENT', 'RT_MRV', 'MRV_MPR', 'SUB_MIP_MRA_TOF_ENT', 'MIP_SUB_TOF_ENT', 'MIP_TOF_ENT', 'FA', 'AAHSCOUT_MPR ', 'MIP ', 'GRE ', 'T1_MIP_TOF', 'MIP_T1_MPRAGE', 'MPR_TOF', 'TOF_TOF', 'FLAIR_MPR_MPR', 'BRAVO', 'MPR_FLAIR']


# Delete rows containing strings from the list
df = df[~df['Calculation Result'].isin(strings_to_delete)]

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
}

# Apply the replacement rules using .replace() with regex=True
#run it twice if needed

df['Calculation Result'] = df['Calculation Result'].replace(replacement_rules, regex=True)
df.reset_index(drop=True, inplace=True)
