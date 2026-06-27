"""
Calculate readability metrics (Flesch-Kincaid, SMOG) for all text versions.
This is part of the evaluation process of this project.
This script will measure how readable the texts are based on the two metrics: Flesch-Kincaid and SMOG
What the metrics mean:
- Flesch-Kincaid:
    - it is a metric used to assess how easy a text is to read
    - The metric score is on a a scale from 0 to 100
    - Meaning of the score:
        - 90->100: Very easy to read, the reading level of an 11 year old
        - 60->89: Standard English level, the level of the average teenager or adult
        - 30->59: Difficult to read, collage level reading level
        - 0->29: Very Difficult to read, best suited for professional audiences
-SMOG:
    - evaluates how difficult a text is by measuring the number of polysyllabic words
    -Meaning of the score:
        - 1-6: Highly accessible. This is the score recommended for the general audiences 
        - 7-9: Ideal for the average reader
        - 9-12: Highschool reading level
        - 13+ : Academic level
"""

import pandas as pd
from textstat import flesch_kincaid_grade, smog_index
import nltk
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
print("Downloading NLTK data...")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def calculate_readability(text):
    """This function computes the readability metrics for the given texts
    The returns of this function are the following: the FK score, the SMOG score, the average sentence length and average word length
    """
    if pd.isna(text) or str(text).strip() == "": 
        return {
            'fk': None,
            'smog': None,
            'avg_sent_len': None,
            'avg_word_len': None
        }
    #This checks for empty or null values
    try:
        text = str(text)
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        # Filter out punctuation from words
        words = [w for w in words if w.isalnum()]
        
        return {
            'fk': flesch_kincaid_grade(text),
            'smog': smog_index(text),
            'avg_sent_len': len(words) / len(sentences) if len(sentences) > 0 else 0,
            'avg_word_len': sum(len(w) for w in words) / len(words) if len(words) > 0 else 0
        }
    except:
        return {'fk': None, 'smog': None, 'avg_sent_len': None, 'avg_word_len': None}

def interpret_fk(score):
    """Interpret Flesch-Kincaid score"""
    if score is None:
        return "N/A"
    if score >= 90:
        return "Very easy (11-year-old)"
    elif score >= 60:
        return "Standard (teen/adult)"
    elif score >= 30:
        return "Difficult (college)"
    else:
        return "Very difficult (academic)"

def interpret_smog(score):
    """Interpret SMOG score"""
    if score is None:
        return "N/A"
    if score <= 6:
        return "Highly accessible"
    elif score <= 9:
        return "Ideal for average reader"
    elif score <= 12:
        return "High school level"
    else:
        return "Academic level"

print("="*70)
print("CALCULATING READABILITY METRICS")
print("="*70)

# Load master dataset
print("\nLoading dataset...")
df = pd.read_csv(
    'Data/Merging_Files/dataset_all_adaptations.csv',
    encoding='utf-8-sig'
)
print(f"Found {len(df)} texts")

# Text columns to process
text_columns = [
    'CuratorialText',
    'lexical_adapted',
    'persona1_adapted',
    'persona2_adapted',
    'persona3_adapted'
]

# Calculate metrics for each column
for col in text_columns:
    if col not in df.columns:
        print(f" Skipping {col} - column not found")
        continue
        
    print(f"\nProcessing {col}...")
    
    metrics_list = []
    for text in tqdm(df[col], desc=f"  {col}"):
        metrics_list.append(calculate_readability(text))
    
    # Add metrics as new columns
    df[f'{col}_fk'] = [m['fk'] for m in metrics_list]
    df[f'{col}_smog'] = [m['smog'] for m in metrics_list]
    df[f'{col}_avg_sent_len'] = [m['avg_sent_len'] for m in metrics_list]
    df[f'{col}_avg_word_len'] = [m['avg_word_len'] for m in metrics_list]

# Save enhanced dataset
print("\nSaving dataset with readability metrics...")
df.to_csv('Data/Readability_Results/dataset_with_readability.csv', index=False, encoding='utf-8-sig')

# ==================== NEW: SUMMARY STATISTICS ====================

print("\n" + "=" * 70)
print("SUMMARY: AVERAGE READABILITY METRICS BY ADAPTATION")
print("=" * 70)

summary_data = []

for col in text_columns:
    if col not in df.columns:
        continue
    
    # Get the metric columns
    fk_col = f'{col}_fk'
    smog_col = f'{col}_smog'
    avg_sent_col = f'{col}_avg_sent_len'
    avg_word_col = f'{col}_avg_word_len'
    
    # Calculate averages (exclude NaN values)
    avg_fk = df[fk_col].mean()
    avg_smog = df[smog_col].mean()
    avg_sent_len = df[avg_sent_col].mean()
    avg_word_len = df[avg_word_col].mean()
    
    # Format column name nicely
    col_display = col.replace('_adapted', '').replace('_', ' ').title()
    if col == 'CuratorialText':
        col_display = 'Original'
    
    summary_data.append({
        'Text Type': col_display,
        'FK Grade': f"{avg_fk:.2f}",
        'FK Interpretation': interpret_fk(avg_fk),
        'SMOG Index': f"{avg_smog:.2f}",
        'SMOG Interpretation': interpret_smog(avg_smog),
        'Avg Sent Len': f"{avg_sent_len:.2f}",
        'Avg Word Len': f"{avg_word_len:.2f}"
    })

summary_df = pd.DataFrame(summary_data)

print("\n")
print(summary_df.to_string(index=False))

# Save summary to CSV
import os
output_dir = 'Data/Readability_Results'
os.makedirs(output_dir, exist_ok=True)

summary_path = f'{output_dir}/readability_metrics_summary.csv'
summary_df.to_csv(summary_path, index=False)
print(f"\n Summary saved to: {summary_path}")

print("\n" + "="*70)
print("READABILITY METRICS COMPLETE")
print("="*70)
print(f" All metrics calculated")
print(f" Full dataset saved to: Data/Readability_Results/dataset_with_readability.csv")
print(f" Summary statistics saved to: {summary_path}")
print("="*70)
"""
Results of the metrics after a successful run:
Text Type    FK Grade   FK Interpretation      SMOG Index   SMOG Interpretation Avg Sent Len Avg Word Len
 Original    12.30 Very difficult (academic)      13.64      Academic level        21.95         5.03
  Lexical    11.14 Very difficult (academic)      12.22      Academic level        22.14         4.78
 Persona1     7.75 Very difficult (academic)       9.87   High school level        16.05         4.52
 Persona2    11.05 Very difficult (academic)      13.03      Academic level        18.68         5.01
 Persona3     9.09 Very difficult (academic)      10.98   High school level        18.13         4.65


"""