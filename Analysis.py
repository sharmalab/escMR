import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
pd.options.display.max_rows = 4000

df = pd.read_csv('')
df= df.drop_duplicates(subset='SeriesInstanceUID')
df.reset_index(inplace=True)
df = df.drop('index', axis=1)
df['SeriesDescription'] =df['SeriesDescription'].str.upper()
df['SeriesInstanceUID'].nunique()
df['StudyInstanceUID'].nunique()
empty_rows = df[df['SeriesDescription'].isna() | df['SeriesDescription'].isnull()]

if not empty_rows.empty:
    print("Empty rows found:")

else:
    print("No empty rows found.")
empty_rows

df['SeriesDescription'] = df['SeriesDescription'].str.upper()
frequency_counts = df['SeriesDescription'].value_counts()
filtered_frequency_counts = frequency_counts[frequency_counts > 1]

print(filtered_frequency_counts)
total_frequency_sum = filtered_frequency_counts.sum()

print("Total frequency sum:", total_frequency_sum)

selected_column = df["SeriesDescription"]

tokenized_column = selected_column.str.split(r'[\s_()\[\]]+')

# Display the tokenized column
print(tokenized_column)
# Flatten the series of lists into a single list
tokenized_column_flat = tokenized_column.explode()

df['StudyDescription'] =df['StudyDescription'].str.upper()
frequency_cou = df['StudyDescription'].value_counts()
filtered_frequency_cou = frequency_cou[frequency_cou > 0]
filtered_frequency_cou.sort_index(inplace = True)
print(filtered_frequency_cou)
total_frequency_ = filtered_frequency_cou.sum()

print("Total frequency sum:", total_frequency_)

output_file = os.path.join(directory, 'filtered_frequency_cou.csv')
filtered_frequency_cou.to_csv(output_file, header=True)


print(f"Saved merged_df_unique as CSV file: {output_file}")


# Get unique tokens
unique_tokens = tokenized_column_flat.unique()

# Display the unique tokens
print(unique_tokens)


# Count the frequency of each token
token_counts= tokenized_column_flat.value_counts()

# Display the token counts
print(token_counts)
