
## Labelling

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

These instructions are presented in a clear and organized format for your README file.
