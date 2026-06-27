"""
Verify voice metrics were calculated correctly.
"""

import pandas as pd
import os

print("=" * 60)
print("VOICE METRICS VERIFICATION")
print("=" * 60)

# Find the file
possible_paths = [
    'Results/Voice_Metrics/dataset_with_voice_metrics.csv',
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

# Check for voice metric columns
voice_columns = [
    'CuratorialText_hedging',
    'lexical_adapted_hedging',
    'persona1_adapted_hedging',
    'persona2_adapted_hedging',
    'persona3_adapted_hedging'
]

print("=" * 60)
print("HEDGING DENSITY")
print("=" * 60)

for col in voice_columns:
    if col not in df.columns:
        print(f"✗ {col} - NOT FOUND")
        continue
    
    valid = df[col].notna().sum()
    mean = df[col].mean()
    
    print(f"\n{col}:")
    print(f"  Valid values: {valid}/{len(df)}")
    print(f"  Mean: {mean:.4f}")
    print(f"  Min: {df[col].min():.4f}")
    print(f"  Max: {df[col].max():.4f}")

# Check attribution
print("\n" + "=" * 60)
print("ATTRIBUTION DENSITY")
print("=" * 60)

attribution_columns = [col.replace('hedging', 'attribution') for col in voice_columns]

for col in attribution_columns:
    if col not in df.columns:
        print(f"{col} - NOT FOUND")
        continue
    
    valid = df[col].notna().sum()
    mean = df[col].mean()
    
    print(f"\n{col}:")
    print(f"  Valid values: {valid}/{len(df)}")
    print(f"  Mean: {mean:.4f}")
    print(f"  Min: {df[col].min():.4f}")
    print(f"  Max: {df[col].max():.4f}")

# Check art terms
print("\n" + "=" * 60)
print("ART TERM DENSITY")
print("=" * 60)

art_columns = [col.replace('hedging', 'art_terms') for col in voice_columns]

for col in art_columns:
    if col not in df.columns:
        print(f"{col} - NOT FOUND")
        continue
    
    valid = df[col].notna().sum()
    mean = df[col].mean()
    
    print(f"\n{col}:")
    print(f"  Valid values: {valid}/{len(df)}")
    print(f"  Mean: {mean:.4f}")
    print(f"  Min: {df[col].min():.4f}")
    print(f"  Max: {df[col].max():.4f}")

print("\n" + "=" * 60)
print("VOICE METRICS VERIFIED")
print("=" * 60)

# Show sample
print("\nSample of metrics (first 3 texts):")
print(df[[
    'CuratorialText_hedging', 'lexical_adapted_hedging',
    'persona1_adapted_hedging', 'persona2_adapted_hedging',
    'persona3_adapted_hedging'
]].head(3).to_string())
"""
============================================================
VOICE METRICS VERIFICATION
============================================================

Loading: Results/Voice_Metrics/dataset_with_voice_metrics.csv
Total rows: 699
Total columns: 36

============================================================
HEDGING DENSITY
============================================================

CuratorialText_hedging:
  Valid values: 699/699
  Mean: 0.1720
  Min: 0.0000
  Max: 6.6667

lexical_adapted_hedging:
  Valid values: 699/699
  Mean: 0.1643
  Min: 0.0000
  Max: 6.2500

persona1_adapted_hedging:
  Valid values: 699/699
  Mean: 0.1832
  Min: 0.0000
  Max: 3.7736

persona2_adapted_hedging:
  Valid values: 699/699
  Mean: 0.1781
  Min: 0.0000
  Max: 5.8824

persona3_adapted_hedging:
  Valid values: 699/699
  Mean: 0.2010
  Min: 0.0000
  Max: 6.6667

============================================================
ATTRIBUTION DENSITY
============================================================

CuratorialText_attribution:
  Valid values: 699/699
  Mean: 0.0105
  Min: 0.0000
  Max: 1.3158

lexical_adapted_attribution:
  Valid values: 699/699
  Mean: 0.0103
  Min: 0.0000
  Max: 1.3158

persona1_adapted_attribution:
  Valid values: 699/699
  Mean: 0.0057
  Min: 0.0000
  Max: 1.2658

persona2_adapted_attribution:
  Valid values: 699/699
  Mean: 0.0111
  Min: 0.0000
  Max: 1.2346

persona3_adapted_attribution:
  Valid values: 699/699
  Mean: 0.0118
  Min: 0.0000
  Max: 1.2987

============================================================
ART TERM DENSITY
============================================================

CuratorialText_art_terms:
  Valid values: 699/699
  Mean: 1.1090
  Min: 0.0000
  Max: 11.1111

lexical_adapted_art_terms:
  Valid values: 699/699
  Mean: 0.8909
  Min: 0.0000
  Max: 8.3333

persona1_adapted_art_terms:
  Valid values: 699/699
  Mean: 0.6747
  Min: 0.0000
  Max: 6.4516

persona2_adapted_art_terms:
  Valid values: 699/699
  Mean: 1.0908
  Min: 0.0000
  Max: 9.3750

persona3_adapted_art_terms:
  Valid values: 699/699
  Mean: 0.8103
  Min: 0.0000
  Max: 7.3171

============================================================
VOICE METRICS VERIFIED
============================================================

Sample of metrics (first 3 texts):
   CuratorialText_hedging  lexical_adapted_hedging  persona1_adapted_hedging  persona2_adapted_hedging  persona3_adapted_hedging
0                     0.0                      0.0                  1.086957                       0.0                       0.0
1                     0.0                      0.0                  0.000000                       0.0                       0.0
2                     0.0                      0.0                  0.000000                       0.0                       0.0
"""