"""
This script calculates the semantic metrics for the adapted curatorial texts.
The following metrics are computed:
1. Semantic Similarit
2. Entity Preservation
3. BERTScore F1
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score as bert_score
import spacy
from tqdm import tqdm
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

print("=" * 70)
print("CALCULATING SEMANTIC METRICS")
print("=" * 70)


#This is a helper function meant to check for NULL values
def is_null(x):
    """Check if value is NULL"""
    if pd.isna(x):
        return True

    if x is None:
        return True

    if isinstance(x, str):

        cleaned = x.strip().upper()

        if cleaned == "":
            return True

        if cleaned == "NULL":
            return True

    return False

# This loads the models
print("\nLoading models (this may take a minute)...")

print("  Loading sentence transformer...")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

print("  Loading spaCy model...")
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("   Installing spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm") # This loads the NLP in English

print("Models loaded\n")


def calculate_semantic_similarity(original_texts, adapted_texts):
    """
    Calculate semantic similarity for all pairs at once (FAST!).
    Returns array with similarities or NaN for NULL pairs.
    """
    
    # Track which indices are valid
    valid_idx = []
    texts_to_encode_orig = []
    texts_to_encode_adapt = []
    
    # First pass: identify valid texts
    for i, (orig, adapt) in enumerate(zip(original_texts, adapted_texts)):
        if not is_null(orig) and not is_null(adapt):
            valid_idx.append(i)
            texts_to_encode_orig.append(str(orig).strip())
            texts_to_encode_adapt.append(str(adapt).strip())
    
    # Initialize result array (all NaN)
    similarities = [np.nan] * len(original_texts)
    
    # If no valid pairs, return NaN array
    if len(valid_idx) == 0:
        return similarities
    
    try:
        # BATCH ENCODE (FAST!) instead of one-by-one
        embeddings_orig = semantic_model.encode(
            texts_to_encode_orig,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        embeddings_adapt = semantic_model.encode(
            texts_to_encode_adapt,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Calculate all similarities at once
        sim_matrix = cosine_similarity(embeddings_orig, embeddings_adapt)
        
        # Extract diagonal (original vs adapted for same index)
        for idx, valid_i in enumerate(valid_idx):
            similarity = float(sim_matrix[idx, idx])
            # Clip to [0, 1]
            similarity = np.clip(similarity, 0.0, 1.0)
            similarities[valid_i] = similarity
        
        return similarities
    
    except Exception as e:
        print(f"Error: {e}")
        return similarities

def calculate_entity_preservation(original_texts, adapted_texts):
    """Calculate entity preservation for all pairs."""
    
    results = []
    
    for orig, adapt in tqdm(
        zip(original_texts, adapted_texts),
        total=len(original_texts),
        desc="    Entity preservation",
        leave=False
    ):
        
        if is_null(orig) or is_null(adapt):
            results.append(np.nan)
            continue
        
        try:
            doc_orig = nlp(str(orig).strip())
            doc_adapt = nlp(str(adapt).strip())
            
            entities_orig = {ent.text.lower() for ent in doc_orig.ents}
            entities_adapt = {ent.text.lower() for ent in doc_adapt.ents}
            
            if len(entities_orig) == 0:
                results.append(np.nan)
            else:
                preservation = len(entities_orig & entities_adapt) / len(entities_orig)
                preservation = np.clip(preservation, 0.0, 1.0)
                results.append(preservation)
        
        except:
            results.append(np.nan)
    
    return results

# The Calculation of BERTScore
def calculate_bertscore(original_texts, adapted_texts):
    """Calculate BERTScore for all pairs efficiently."""
    
    # Identify valid pairs
    valid_idx = []
    clean_orig = []
    clean_adapt = []
    
    for i, (o, a) in enumerate(zip(original_texts, adapted_texts)):
        if not is_null(o) and not is_null(a):
            valid_idx.append(i)
            clean_orig.append(str(o).strip())
            clean_adapt.append(str(a).strip())
    
    # Initialize result array (all NaN)
    scores = [np.nan] * len(original_texts)
    
    # If no valid pairs, return NaN array
    if len(clean_orig) == 0:
        return scores
    
    try:
        # Compute BERTScore with larger batch size
        P, R, F1 = bert_score(
            clean_adapt,
            clean_orig,
            lang="en",
            verbose=False,
            batch_size=16
        )
        
        # Extract F1 scores
        f1_numpy = F1.cpu().numpy()
        
        # Map back to original indices
        for idx, valid_i in enumerate(valid_idx):
            f1_score = float(f1_numpy[idx])
            # Clip to [0, 1]
            f1_score = np.clip(f1_score, 0.0, 1.0)
            scores[valid_i] = f1_score
        
        return scores
    
    except Exception as e:
        print(f"Error: {e}")
        return scores


# Loading the dataset
print("Loading dataset...")
df = pd.read_csv(
    'Data/Merging_Files/dataset_all_adaptations.csv',
    encoding='utf-8-sig'
)

original_col = 'CuratorialText'
if original_col not in df.columns:
    print(f" ERROR: Cannot find '{original_col}' column")
    print(f"  Available columns: {list(df.columns)}")
    exit(1)


adapted_columns = [
    'lexical_adapted',
    'persona1_adapted',
    'persona2_adapted',
    'persona3_adapted'
]

print("Calculating semantic metrics...\n")

for col in adapted_columns:
    
    if col not in df.columns:
        print(f"Skipping '{col}' - column not found")
        continue
    
    print(f"Processing '{col}'...")
    
    original_texts = df[original_col].tolist()
    adapted_texts = df[col].tolist()
    
    # Semantic similarity
    print(f"  Calculating semantic similarity...")
    semantic_sim = calculate_semantic_similarity(original_texts, adapted_texts)
    df[f'{col}_semantic_sim'] = semantic_sim
    valid_sim = sum(1 for s in semantic_sim if not np.isnan(s))
    print(f"    {valid_sim}/{len(semantic_sim)} valid scores")
    
    # Entity preservation
    print(f"  Calculating entity preservation...")
    entity_pres = calculate_entity_preservation(original_texts, adapted_texts)
    df[f'{col}_entity_preservation'] = entity_pres
    valid_ent = sum(1 for e in entity_pres if not np.isnan(e))
    print(f"    {valid_ent}/{len(entity_pres)} valid scores")
    
    # BERTScore
    print(f"  Calculating BERTScore...")
    bertscores = calculate_bertscore(original_texts, adapted_texts)
    df[f'{col}_bertscore_f1'] = bertscores
    valid_bert = sum(1 for b in bertscores if not np.isnan(b))
    print(f"    {valid_bert}/{len(bertscores)} valid scores")
    
    print(f"  '{col}' complete\n")

# Saving the output
output_dir = 'Data/Semantic_Metrics'
os.makedirs(output_dir, exist_ok=True)

output_path = f'{output_dir}/dataset_with_semantic_metrics.csv'

print("Saving dataset with all metrics...")
df.to_csv(
    output_path,
    index=False,
    encoding='utf-8-sig',
    na_rep=''  # Keeps blanks clean for Excel
)


# Summary of the run
print("\n" + "=" * 70)
print("SUMMARY: AVERAGE SEMANTIC METRICS BY ADAPTATION")
print("=" * 70)

summary_data = []

for col in adapted_columns:
    sim_col = f'{col}_semantic_sim'
    ent_col = f'{col}_entity_preservation'
    bert_col = f'{col}_bertscore_f1'
    
    # Calculate averages
    avg_sim = df[sim_col].mean() if sim_col in df.columns else np.nan
    avg_ent = df[ent_col].mean() if ent_col in df.columns else np.nan
    avg_bert = df[bert_col].mean() if bert_col in df.columns else np.nan
    
    # Calculate composite semantic score (average of all three)
    composite = (avg_sim + avg_ent + avg_bert) / 3 if not np.isnan(avg_sim) else np.nan
    
    # Format column name nicely
    col_display = col.replace('_adapted', '').replace('_', ' ').title()
    
    summary_data.append({
        'Text Type': col_display,
        'Semantic Sim': f"{avg_sim:.4f}",
        'Entity Pres': f"{avg_ent:.4f}",
        'BERTScore F1': f"{avg_bert:.4f}",
        'Composite': f"{composite:.4f}"
    })

summary_df = pd.DataFrame(summary_data)

print("\n")
print(summary_df.to_string(index=False))

# Calculate % change from original
print("\n" + "=" * 70)
print("SEMANTIC PRESERVATION BY ADAPTATION")
print("=" * 70)

original_composite = None
for _, row in summary_df.iterrows():
    if row['Text Type'] == 'Lexical Adapted':  # Get first adaptation's baseline
        # Find original in a different way - just take first row average
        break

# Calculate original baseline
original_sim = df['CuratorialText_semantic_sim'].mean() if 'CuratorialText_semantic_sim' in df.columns else np.nan

print("\nAverage semantic metrics compared to original:\n")

for col in adapted_columns:
    col_display = col.replace('_adapted', '').replace('_', ' ').title()
    sim_col = f'{col}_semantic_sim'
    
    if sim_col in df.columns:
        adapted_sim = df[sim_col].mean()
        
        if not np.isnan(original_sim) and not np.isnan(adapted_sim):
            percent_change = ((adapted_sim - original_sim) / original_sim) * 100
            direction = '↓ Loss' if percent_change < 0 else '↑ Gain'
            print(f"  {col_display:15} Semantic Sim: {percent_change:+.2f}%  {direction}")

# Save summary to CSV
summary_path = f'{output_dir}/semantic_metrics_summary.csv'
summary_df.to_csv(summary_path, index=False)
print(f"\n Summary saved to: {summary_path}")

print("\n" + "=" * 70)
print("DETAILED METRIC BREAKDOWN")
print("=" * 70)

for col in adapted_columns:
    print(f"\n{col}:")
    
    sim_col = f'{col}_semantic_sim'
    if sim_col in df.columns:
        valid = df[sim_col].notna().sum()
        mean = df[sim_col].mean()
        std = df[sim_col].std()
        min_val = df[sim_col].min()
        max_val = df[sim_col].max()
        print(f"  Semantic Similarity:")
        print(f"    Valid:  {valid}/{len(df)}")
        print(f"    Mean:   {mean:.4f}, SD: {std:.4f}")
        print(f"    Range:  [{min_val:.4f}, {max_val:.4f}]")
    
    ent_col = f'{col}_entity_preservation'
    if ent_col in df.columns:
        valid = df[ent_col].notna().sum()
        mean = df[ent_col].mean()
        std = df[ent_col].std()
        min_val = df[ent_col].min()
        max_val = df[ent_col].max()
        print(f"  Entity Preservation:")
        print(f"    Valid:  {valid}/{len(df)}")
        print(f"    Mean:   {mean:.4f}, SD: {std:.4f}")
        print(f"    Range:  [{min_val:.4f}, {max_val:.4f}]")
    
    bert_col = f'{col}_bertscore_f1'
    if bert_col in df.columns:
        valid = df[bert_col].notna().sum()
        mean = df[bert_col].mean()
        std = df[bert_col].std()
        min_val = df[bert_col].min()
        max_val = df[bert_col].max()
        print(f"  BERTScore F1:")
        print(f"    Valid:  {valid}/{len(df)}")
        print(f"    Mean:   {mean:.4f}, SD: {std:.4f}")
        print(f"    Range:  [{min_val:.4f}, {max_val:.4f}]")

print("\n" + "=" * 70)
print("SEMANTIC METRICS COMPLETE")
print("=" * 70)
print(f" Dataset: {len(df)} rows × {len(df.columns)} columns")
print(f" Full dataset saved to: {output_path}")
print(f" Summary statistics saved to: {summary_path}")
print("=" * 70)
