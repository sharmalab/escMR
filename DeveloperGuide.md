## **Retraining the Model with New Sequences**

If you need to retrain the model to predict the new sequences (weightings), follow these steps:
### Change the path  to CSV file accordingly:

df = pd.read_csv(path/to/your/new/data.csv')
Replace 'path/to/your/new/data.csv' with the actual file path.

### Modify the Labelling Part of Training.py:

### Edit the 'tokens_to_search' List

To include new weighting types, list only those tokens that can help distinguish sequences from others in the study. You do not need to include anatomical plane information. Focus on keywords that determine weighting:

```python
# Modify the tokens_to_search list by adding or removing tokens as needed. For example:
tokens_to_search = ['T1', 'T2', 'SWI', 'NewToken1', 'NewToken2']
```

### Edit the Replacement Rules

Edit the dictionary of replacement rules:

```python
# Edit the dictionary
replacement_rules = {
   'incorrect form': 'correct form',
}
```

### Edit the List of Strings to Be Deleted

If you encounter issues with certain results, edit the list of strings to be deleted:

```python
strings_to_delete = []
```


---
### Modifying Code to Filter Rows with Target Strings

If you want to modify the given code to filter rows containing specific target strings in a DataFrame, follow these steps:

1. **Define Your List of Target Strings:**

   Begin by creating a list of strings that you want to use as target strings. For example, you can have a list like this:

   ```python
   target_strings = ['DWI', 'T2', 'FLAIR', 'T2*', 'T1', 'SCOUT', 'VIBE', 'CISS', 'TOF', 'DIR_SPACE', 'T2_SPACE', 'PERF', 'DTI', 'FGATIR', 'T1_FLAIR', 'MRV', 'FIESTA', 'T1_MPRAGE', 'MRA', .....]


### Modifying Code to Filter Rows with Target Strings

If you want to modify the given code to filter rows containing specific target strings in a DataFrame, follow these steps:

1. **Define Your List of Target Strings:**

   Begin by creating a list of strings that you want to use as target strings. For example, you can have a list like this:

   ```python
   target_strings = ['DWI', 'T2', 'FLAIR', 'T2*', 'T1', 'SCOUT', 'VIBE', 'CISS', 'TOF', 'DIR_SPACE', 'T2_SPACE', 'PERF', 'DTI', 'FGATIR', 'T1_FLAIR', 'MRV', 'FIESTA', 'T1_MPRAGE', 'MRA']


### Modify Your DataFrame (Assumed as 'df'):

Assuming you have a DataFrame named 'df', you can use the following code to filter rows based on your target strings:

```python
filtered_rows = df[df.apply(lambda row: any(any(target in str(cell) for target in target_strings) if isinstance(cell, str) else False for cell in row), axis=1)]
```

This code filters rows in 'df' containing any target strings.

### 'filtered_rows' Contains the Filtered Rows:

The variable 'filtered_rows' now contains the rows that match your criteria.

## Updating default criteria dictionary in Code.py


The updated criteria dictionary you want to create consists of two parts: the criteria for sequences ('PRESENCE' and 'LENGTH') and the details for specific sequence protocols (eg 'DWIFS2DAX','T1POST2DSAG','T1POST3DSAG' etc). Instructions for naming sequences are present in README.md
Let's break it down step by step:

### 1. Define Criteria for Sequences

Start by specifying the criteria for protocol. These criteria include sequences that need to be present ('PRESENCE') and the acceptable length range ('LENGTH') for those sequences.

```python
# Example
criteria = {'PROTOCOLNAME': {
    'PRESENCE': {'DWI2DAX', 'FLAIR2DAX','T12DSAG','SWIMIP3DAX','T12DAX','SEQUENCENAME'},
    'LENGTH': {X, Y}
}}
```
- `'PRESENCE'` lists the sequences that should be present in this protocol.
- `'LENGTH'` specifies the acceptable range of sequence lengths.

### 2. Define Specific Sequence Protocols

Now, let's define the details for specific sequence of protocols. Each sequence should include orientation, FOV, pixel area, thickness, gap, and coverage.

#### For eg 'SEQUENCENAME':

```python
'SEQUENCENAME': {
    'Orientation': '2D',
    'FOV': [140, 180],
    'PixelArea': 0.6,
    'Thickness': 3,
    'Gap': 0.5,
    'Coverage': 50
}
```

- `'Orientation'`: Specify whether it's a '2D' or '3D' sequence.
- `'FOV'`: Define the FOV range as a list with two values.
- `'PixelArea'`: Provide the pixel area as a floating-point value.
- `'Thickness'`: Enter the thickness as a floating-point value.
- `'Gap'`: Set the gap value as a floating-point number.
- `'Coverage'`: Specify the coverage as a floating-point value.

### Combine the criteria for sequences and the details for specific sequence protocols in the criteria dictionary :

```python
criteria = {
    'PROTOCOLNAME': {
        'PRESENCE': {'DWIFS2DAX', 'T12DAX', 'SEQUENCENAME'},
        'LENGTH': {X, Y}
    },
    'SEQUENCENAME': {
        'Orientation': '2D',
        'FOV': [140, 180],
        'PixelArea': 0.6,
        'Thickness': 3,
        'Gap': 0.5,
        'Coverage': 50
    },
.
.
.
```

Repeat this structure for each specific protocol you want to include in your criteria dictionary.

---
