"""
Validate that your dataset has all required columns and proper formatting.
Run this FIRST before anything else.
"""
import pandas as pd
import os

print("="*60)
print("DATASET VALIDATION")
print("="*60)

# Load dataset
try: 
    df = pd.read_excel('Data/Original/Dataset_Curatorial_Text.xlsx')
    print(f"Dataset loaded: {len(df)} texts found")
except FileNotFoundError:
    print("ERROR: dataset_original.csv not found in data/original/")
    print("  Please place your dataset there and try again.")
    exit(1)

# Required columns
required_columns = [
    'ID',           # Unique identifier
    'ReferenceNumber', #The reference number of the artwork from the museum
    'Museum',       # Name of the museum
    'NameOfTheObject',         # The name of the artwork
    'TextType',        # Exhibition/Wall/Section/Object label
    'CuratorialText',     # The actual curatorial text
    'WordCountCuratorialText',      # Nr words of the curatorial text
]

print("\nChecking required columns...")
missing = []
for col in required_columns:
    if col in df.columns:
        print(f"  {col}")
    else:
        print(f"  {col} - MISSING")
        missing.append(col)

if missing:
    print(f"\n ERROR: Missing columns: {missing}")
    print("  Please add these columns to your dataset.")
    exit(1)

# Check for empty values in original_text
empty_texts = df['CuratorialText'].isna().sum()
if empty_texts > 0:
    print(f"\n WARNING: {empty_texts} texts have empty 'original_text'")
    print("  These will be skipped during processing.")

# Check text length distribution
print(f"\nText Length Statistics:")
print(f"  Mean: {df['WordCountCuratorialText'].mean():.1f} words")
print(f"  Median: {df['WordCountCuratorialText'].median():.1f} words")
print(f"  Min: {df['WordCountCuratorialText'].min()} words")
print(f"  Max: {df['WordCountCuratorialText'].max()} words")

# Check text types
print(f"\nText Type Distribution:")
print(df['TextType'].value_counts())

print("\n" + "="*60)
print("VALIDATION COMPLETE - Dataset is ready for processing!")
print("="*60)

"""
- for the first few runs I had an issue connecting the key to the code
- also the initial issue was not being able to connect to deepseek API
- but it worked, after fixing imports and realising a misspelling in the key name
- the key was names key_1 instead of DEEPSEEK_API_KEY
- this is the output of the code after the fixes, which shows that the dataset is valid
and ready for provessing
Results from validation:
============================================================
DATASET VALIDATION
============================================================
Dataset loaded: 699 texts found

Checking required columns...
   ID
   ReferenceNumber
   Museum
   NameOfTheObject
   TextType
   CuratorialText
   WordCountCuratorialText

Text Length Statistics:
  Mean: 101.7 words
  Median: 84.0 words
  Min: 10 words
  Max: 923 words

Text Type Distribution:
TextType
ObjectLabel       360
SectionText       210
ExhibitionText     90
WallText           39
Name: count, dtype: int64

============================================================
VALIDATION COMPLETE - Dataset is ready for processing!
============================================================
"""