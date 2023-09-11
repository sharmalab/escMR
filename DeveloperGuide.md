## **Retraining the Model with New Sequences**

If you need to retrain the model to predict the new sequences (weightings), follow these steps:
### Change the path  to CSV file accordingly:

df = pd.read_csv(path/to/your/new/data.csv')
Replace 'path/to/your/new/data.csv' with the actual file path.

## Modify the Labelling Part of Training.py:

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

This code filters rows in 'df' that contain any of the target strings.

### 'filtered_rows' Contains the Filtered Rows:

The variable 'filtered_rows' now contains the rows that match your criteria.

