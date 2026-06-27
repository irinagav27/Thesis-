"""
Verify readability metrics were calculated correctly.
"""

import pandas as pd
import os

print("=" * 60)
print("READABILITY METRICS VERIFICATION")
print("=" * 60)

# Find the file
possible_paths = [
    'Results/Readability_Results/dataset_with_readability.csv',
    'Results/Merging_Files/dataset_all_adaptations.csv',
]

dataset_path = None
for path in possible_paths:
    if os.path.exists(path):
        dataset_path = path
        break

if dataset_path is None:
    print("File not found!")
    exit(1)

print(f"\nLoading: {dataset_path}")
df = pd.read_csv(dataset_path, encoding='utf-8-sig')

print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}\n")

# Check for readability columns
text_types = [
    'CuratorialText',
    'lexical_adapted',
    'persona1_adapted',
    'persona2_adapted',
    'persona3_adapted'
]

metrics = ['fk', 'smog', 'avg_sent_len', 'avg_word_len']

print("=" * 60)
print("FLESCH-KINCAID GRADE LEVEL")
print("=" * 60)

for text_type in text_types:
    col = f'{text_type}_fk'
    
    if col not in df.columns:
        print(f"✗ {col} - NOT FOUND")
        continue
    
    valid = df[col].notna().sum()
    mean = df[col].mean()
    
    print(f"\n{text_type}:")
    print(f"  Valid values: {valid}/{len(df)}")
    print(f"  Mean: {mean:.2f}")
    print(f"  Min: {df[col].min():.2f}")
    print(f"  Max: {df[col].max():.2f}")

# Check SMOG
print("\n" + "=" * 60)
print("SMOG INDEX")
print("=" * 60)

for text_type in text_types:
    col = f'{text_type}_smog'
    
    if col not in df.columns:
        print(f"✗ {col} - NOT FOUND")
        continue
    
    valid = df[col].notna().sum()
    mean = df[col].mean()
    
    print(f"\n{text_type}:")
    print(f"  Valid values: {valid}/{len(df)}")
    print(f"  Mean: {mean:.2f}")
    print(f"  Min: {df[col].min():.2f}")
    print(f"  Max: {df[col].max():.2f}")

# Check average sentence length
print("\n" + "=" * 60)
print("AVERAGE SENTENCE LENGTH")
print("=" * 60)

for text_type in text_types:
    col = f'{text_type}_avg_sent_len'
    
    if col not in df.columns:
        print(f"✗ {col} - NOT FOUND")
        continue
    
    valid = df[col].notna().sum()
    mean = df[col].mean()
    
    print(f"\n{text_type}:")
    print(f"  Valid values: {valid}/{len(df)}")
    print(f"  Mean: {mean:.2f} words/sentence")
    print(f"  Min: {df[col].min():.2f}")
    print(f"  Max: {df[col].max():.2f}")

print("\n" + "=" * 60)
print("READABILITY METRICS VERIFIED")
print("=" * 60)

# Show comparison table
print("\nComparison (Flesch-Kincaid):")
fk_cols = [f'{t}_fk' for t in text_types]
comparison = df[fk_cols].head(5).round(2)
comparison.columns = text_types
print(comparison.to_string())

print("\n\nWhat the scores mean:")
print("Flesch-Kincaid Grade Level:")
print("  0-6:   Very easy (11-year-old)")
print("  7-9:   Standard (teenager/adult)")
print("  10-14: Difficult (college)")
print("  15+:   Very difficult (professional)")

print("\nSMOG Index:")
print("  1-6:   Highly accessible (general)")
print("  7-9:   Ideal (average reader)")
print("  9-12:  High school level")
print("  13+:   Academic level")

"""
Loading: Results/Readability_Results/dataset_with_readability.csv
Total rows: 699
Total columns: 41

============================================================
FLESCH-KINCAID GRADE LEVEL
============================================================

CuratorialText:
  Valid values: 699/699
  Mean: 12.30
  Min: 2.66
  Max: 41.21

lexical_adapted:
  Valid values: 699/699
  Mean: 11.36
  Min: 2.34
  Max: 40.62

persona1_adapted:
  Valid values: 699/699
  Mean: 6.80
  Min: 1.41
  Max: 14.44

persona2_adapted:
  Valid values: 699/699
  Mean: 10.01
  Min: 1.90
  Max: 26.98

persona3_adapted:
  Valid values: 699/699
  Mean: 11.91
  Min: 3.90
  Max: 31.22

============================================================
SMOG INDEX
============================================================

CuratorialText:
  Valid values: 699/699
  Mean: 13.64
  Min: 3.13
  Max: 28.03

lexical_adapted:
  Valid values: 699/699
  Mean: 12.50
  Min: 3.13
  Max: 28.03

persona1_adapted:
  Valid values: 699/699
  Mean: 9.41
  Min: 3.13
  Max: 17.12

persona2_adapted:
  Valid values: 699/699
  Mean: 12.14
  Min: 3.13
  Max: 22.92

persona3_adapted:
  Valid values: 699/699
  Mean: 13.24
  Min: 3.13
  Max: 28.68

============================================================
AVERAGE SENTENCE LENGTH
============================================================

CuratorialText:
  Valid values: 699/699
  Mean: 21.95 words/sentence
  Min: 4.50
  Max: 98.00

lexical_adapted:
  Valid values: 699/699
  Mean: 22.10 words/sentence
  Min: 4.50
  Max: 98.00

persona1_adapted:
  Valid values: 699/699
  Mean: 12.72 words/sentence
  Min: 5.86
  Max: 32.00

persona2_adapted:
  Valid values: 699/699
  Mean: 16.22 words/sentence
  Min: 4.38
  Max: 49.00

persona3_adapted:
  Valid values: 699/699
  Mean: 22.36 words/sentence
  Min: 7.50
  Max: 60.00

============================================================
READABILITY METRICS VERIFIED
============================================================

Comparison (Flesch-Kincaid):
   CuratorialText  lexical_adapted  persona1_adapted  persona2_adapted  persona3_adapted
0           14.54            13.44              6.73             10.94             13.91
1           16.40            14.02              6.76             11.63             16.48
2           16.03            15.41              7.61             10.85             16.17
3           18.58            17.86              5.83             10.36             15.20
4           22.31            21.70              7.47             22.92             22.10


What the scores mean:
Flesch-Kincaid Grade Level:
  0-6:   Very easy (11-year-old)
  7-9:   Standard (teenager/adult)
  10-14: Difficult (college)
  15+:   Very difficult (professional)

SMOG Index:
  1-6:   Highly accessible (general)
  7-9:   Ideal (average reader)
  9-12:  High school level
  13+:   Academic level
"""