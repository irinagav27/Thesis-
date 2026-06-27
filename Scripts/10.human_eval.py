"""
Script 18: Human Evaluation Sample Generator
Load pre-made adaptations and generate sample for human evaluation.
Select strategy, choose number of texts, export to Excel.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# For Excel export
try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
except ImportError:
    print("Installing openpyxl...")
    os.system("pip install openpyxl")
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

print("=" * 70)
print("HUMAN EVALUATION SAMPLE GENERATOR")
print("=" * 70)


# Load the adaptations
print("\nLoading pre-made adaptations...\n")

# Try multiple possible paths
possible_paths = [
    'Data/Merging_Files/dataset_all_adaptations.csv']

df_master = None
loaded_path = None

for path in possible_paths:
    if os.path.exists(path):
        try:
            df_master = pd.read_csv(path, encoding='utf-8-sig')
            loaded_path = path
            print(f" Loaded adaptations from: {path}")
            break
        except Exception as e:
            print(f" Error loading {path}: {e}")
            continue

if df_master is None:
    print("\n  ERROR: Cannot find pre-made adaptations file")
    print("\nPlease ensure you have run:")
    print("  1. Scripts/1.lexical_adaptation.py")
    print("  2. Scripts/2.persona_1_adaptation.py")
    print("  3. Scripts/3.persona_2_adaptation.py")
    print("  4. Scripts/4.persona_3_adaptation.py")
    print("  5. Scripts/5.combine_adaptations.py")
    print("\nOr ensure the combined file exists at one of:")
    for path in possible_paths:
        print(f"  - {path}")
    sys.exit(1)

print(f"  Total texts available: {len(df_master)}\n")


print("=" * 70)
print("STRATEGY SELECTION")
print("=" * 70)

strategies = {
    "1": "lexical",
    "2": "persona1",
    "3": "persona2",
    "4": "persona3",
    "all": "all"
}

print("\nAvailable strategies:")
print("  [1] Lexical Simplification")
print("  [2] Persona 1 (8th-Grade English)")
print("  [3] Persona 2 (Non-Native Expert)")
print("  [4] Persona 3 (Native Non-Expert)")
print("  [all] All Strategies Combined\n")

while True:
    choice = input("Select strategy (1-4 or all): ").strip().lower()
    
    if choice not in strategies:
        print(" Invalid choice. Please enter 1, 2, 3, 4, or all.")
        continue
    
    break

selected_strategy = strategies[choice]
print(f"\n Selected: {selected_strategy.upper()}\n")


# Function to help you choose how many texts you want for the human evaulation
print("=" * 70)
print("SAMPLE SIZE SELECTION")
print("=" * 70)

print(f"\nAvailable texts in dataset: {len(df_master)}")
print("\nOptions:")
print("  [Enter a number, that is an integer between 1 and 699] (e.g., 5, 10, 20, 50)")
print("  [all] Evaluate all texts\n")

while True:
    size_choice = input("How many texts do you want to evaluate? ").strip().lower()
    
    if size_choice == "all":
        sample_size = len(df_master)
        print(f"\n Selected: ALL {sample_size} texts\n")
        break
    
    try:
        sample_size = int(size_choice)
        
        if sample_size < 1:
            print(f" Please enter a number >= 1")
            continue
        
        if sample_size > len(df_master):
            print(f" Only {len(df_master)} texts available. Please enter a smaller number.")
            continue
        
        print(f"\n Selected: {sample_size} texts\n")
        break
    
    except ValueError:
        print(" Invalid input. Please enter a number or 'all'.")
        continue



print("=" * 70)
print("SAMPLE SELECTION")
print("=" * 70)

print(f"\nSelecting {sample_size} texts from dataset...")

# Set random seed for reproducibility
np.random.seed(datetime.now().microsecond)

if sample_size == len(df_master):
    # Use all texts in order
    sample_data = df_master.copy()
    print(f" Using ALL {sample_size} texts\n")
else:
    # Random sample
    sample_indices = np.random.choice(len(df_master), size=sample_size, replace=False)
    sample_data = df_master.iloc[sample_indices].reset_index(drop=True)
    print(f" Selected {sample_size} random texts\n")

print("=" * 70)
print("VERIFYING ADAPTATIONS")
print("=" * 70)

# Determine which strategies to include
if selected_strategy == "all":
    strategies_to_export = ['lexical', 'persona1', 'persona2', 'persona3']
else:
    strategies_to_export = [selected_strategy]

print(f"\nChecking for {len(strategies_to_export)} strategy(ies)...\n")

for strategy in strategies_to_export:
    adapted_col = f'{strategy}_adapted'
    
    if adapted_col not in df_master.columns:
        print(f" ERROR: '{adapted_col}' column not found")
        print(f"  Ensure {strategy} adaptation was generated")
        sys.exit(1)
    
    valid_count = sample_data[adapted_col].notna().sum()
    print(f" {strategy.title()}: {valid_count}/{sample_size} texts available")

print()


print("=" * 70)
print("PREPARING FLAT EXPORT FORMAT")
print("=" * 70)

# Determine strategy column
if selected_strategy == "all":
    print("ERROR: 'all' is not supported in flat format.")
    print("Please select a single strategy (1-4).")
    sys.exit(1)

adapted_col = f"{selected_strategy}_adapted"

if adapted_col not in df_master.columns:
    print(f"ERROR: Column '{adapted_col}' not found")
    sys.exit(1)

# Build dataset rows
export_rows = []

for _, row in sample_data.iterrows():

    if pd.isna(row.get('CuratorialText')):
        continue

    export_rows.append({
        "ID": row.get("ID", row.name),
        "Original Text": row["CuratorialText"],
        "Adapted Text": row.get(adapted_col, ""),
        "Voice Preservation (1-5)": "",
        "Meaning Preservation (1-5)": "",
        "Accessibility (1-5)": "",
        "Comments": ""
    })


# Convert to DataFrame for clean export
df_export = pd.DataFrame(export_rows)



print("Creating Excel file...")

output_dir = "Data/Human_Evaluation"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{output_dir}/human_evaluation_{selected_strategy}_{sample_size}texts_{timestamp}.xlsx"

with pd.ExcelWriter(filename, engine="openpyxl") as writer:
    df_export.to_excel(writer, index=False, sheet_name="Evaluation")

    # Auto-adjust column width (simple)
    worksheet = writer.sheets["Evaluation"]

    for col_idx, col in enumerate(df_export.columns, 1):
        max_len = max(df_export[col].astype(str).map(len).max(), len(col))
        worksheet.column_dimensions[chr(64 + col_idx)].width = min(max_len + 5, 80)

print("=" * 70)
print(" HUMAN EVALUATION FILE CREATED")
print("=" * 70)
print(f"File saved to: {filename}")

print("\nStructure:")
print(" ID")
print(" Original Text")
print(" Adapted Text")
print(" Voice Preservation (1-5)")
print(" Meaning Preservation (1-5)")
print(" Accessibility (1-5)")
print(" Comments")

print("\n" + "=" * 70)