"""
This code is meant to compile a single datasource file from all adaptations.
The resulting file will contain:
- the original curatorial text
- the lexical simplification
- the adaptations for the 3 personas
"""

import pandas as pd
import os

print("=" * 60)
print("COMBINING ALL ADAPTATIONS")
print("=" * 60)

# Loading the original dataset
print("\nLoading original dataset...")
df_original = pd.read_excel(
    'Data/Original/Dataset_Curatorial_Text.xlsx'
)

print(f"  Original: {len(df_original)} texts")

# Loading adaptations
print("\nLoading adaptations...")

df_lexical = pd.read_csv(
    'Data/adapted/Lexical_Simplification/lexical_simplified.csv',
    encoding='utf-8-sig'
)

print(f"  Lexical: {len(df_lexical)} texts")

df_p1 = pd.read_csv(
    'Data/adapted/Persona_1/persona1_adapted.csv',
    encoding='utf-8-sig'
)

print(f"  Persona 1: {len(df_p1)} texts")

df_p2 = pd.read_csv(
    'Data/adapted/Persona_2/persona2_adapted.csv',
    encoding='utf-8-sig'
)

print(f"  Persona 2: {len(df_p2)} texts")

df_p3 = pd.read_csv(
    'Data/adapted/Persona_3/persona3_adapted.csv',
    encoding='utf-8-sig'
)

print(f"  Persona 3: {len(df_p3)} texts")


#Merging everthing in one singular dataset file
print("\nMerging datasets...")

df_master = df_original.copy()

# Lexical simplification
df_master = df_master.merge(
    df_lexical[['ID', 'lexical_adapted']],
    on='ID',
    how='left'
)

# Persona 1
df_master = df_master.merge(
    df_p1[['ID', 'persona1_adapted']],
    on='ID',
    how='left'
)

# Persona 2
df_master = df_master.merge(
    df_p2[['ID', 'persona2_adapted']],
    on='ID',
    how='left'
)

# Persona 3
df_master = df_master.merge(
    df_p3[['ID', 'persona3_adapted']],
    on='ID',
    how='left'
)


# Standardizing the missing values
print("\nStandardizing missing values...")

nullable_columns = [
    'ReferenceNumber',
    'NameOfTheObject',
    'Medium',
    'Dimension',
    'MuseumObjectType',
    'NameCurator'
]

# Convert blank strings to actual missing values
df_master = df_master.replace(
    r'^\s*$',
    pd.NA,
    regex=True
)

# Replace missing values with explicit NULL marker
df_master[nullable_columns] = (
    df_master[nullable_columns]
    .fillna('NULL')
)



os.makedirs(
    'Data/Merging_Files',
    exist_ok=True
)


# Saving to a singular dataset file
output_path = (
    'Data/Merging_Files/'
    'dataset_all_adaptations.csv'
)

print("\nSaving master dataset...")

df_master.to_csv(
    output_path,
    index=False,
    encoding='utf-8-sig'
)


print("\n" + "=" * 60)
print("DATASET COMPILATION OF ALL ADAPTATIONS IS COMPLETE!")
print("=" * 60)

print(f"Total texts: {len(df_master)}")
print(f"Total columns: {len(df_master.columns)}")

print("\nColumns:")
print(list(df_master.columns))

print(f"\nSaved to: {output_path}")

print("=" * 60)
"""
1: After the first run:
* All the text has been successfully merged into a single dataset file.
* The resulting dataset file has the following columns:
['ID', 'ReferenceNumber', 'Museum', 'NameOfTheObject', 'PlaceOfOriginationMuseum',
'PlaceOfOrigination', 'Medium', 'Dimension', 'MuseumObjectType', 
'TextType', 'CuratorialText_x', 'WordCountCuratorialText', 'URL', 
'YearOfTheArtwork', 'NameCurator', 'NameOfTheArtist', 'Language',
'CuratorialText_y', 'persona1_adapted', 'persona2_adapted', 'persona3_adapted']
Total texts: 699
Total columns: 21

2: After the second run
* there was no longer the issue of random characters being insterted
* But, from the original dataset all the values that were marked as NULL are now empty strings
* Next run I will change it as such if a value in the original dataset was null,
that also in the compliation of all adapted texts next to the original texts
and the information from the museums/art pieces is also NULL.

3: After the third run:
* all the empty fields were filled, no more random characters
* all the texts from the inital dataset are present and the adaptations too
"""