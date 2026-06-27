import pandas as pd

print("Loading results...")
df = pd.read_csv('Results\Semantic_Metrics\dataset_with_semantic_metrics.csv')

print("\n" + "="*60)
print("SEMANTIC METRICS VERIFICATION")
print("="*60)

strategies = ['lexical', 'persona1', 'persona2', 'persona3']

for strategy in strategies:
    print(f"\n{strategy.upper()}:")
    
    # Semantic similarity
    sem_col = f'{strategy}_adapted_semantic_sim'
    if sem_col in df.columns:
        valid = df[sem_col].notna().sum()
        mean_score = df[sem_col].mean()
        print(f"  Semantic Similarity:")
        print(f"    Valid scores: {valid}/699")
        print(f"    Mean: {mean_score:.4f}")
        print(f"    Range: {df[sem_col].min():.4f} - {df[sem_col].max():.4f}")
    
    # Entity preservation
    ent_col = f'{strategy}_adapted_entity_preservation'
    if ent_col in df.columns:
        valid = df[ent_col].notna().sum()
        mean_score = df[ent_col].mean()
        print(f"  Entity Preservation:")
        print(f"    Valid scores: {valid}/699")
        print(f"    Mean: {mean_score:.4f}")
        print(f"    Range: {df[ent_col].min():.4f} - {df[ent_col].max():.4f}")
    
    # BERTScore
    bert_col = f'{strategy}_adapted_bertscore_f1'
    if bert_col in df.columns:
        valid = df[bert_col].notna().sum()
        mean_score = df[bert_col].mean()
        print(f"  BERTScore F1:")
        print(f"    Valid scores: {valid}/699")
        print(f"    Mean: {mean_score:.4f}")
        print(f"    Range: {df[bert_col].min():.4f} - {df[bert_col].max():.4f}")

print("\n" + "="*60)
print(" All semantic metrics calculated successfully!")
print("="*60)
"""
LEXICAL:
  Semantic Similarity:
    Valid scores: 699/699
    Mean: 0.9755
    Range: 0.7084 - 1.0000
  Entity Preservation:
    Valid scores: 647/699
    Mean: 0.9698
    Range: 0.0000 - 1.0000
  BERTScore F1:
    Valid scores: 699/699
    Mean: 0.9857
    Range: 0.9408 - 1.0000

PERSONA1:
  Semantic Similarity:
    Valid scores: 699/699
    Mean: 0.9134
    Range: 0.6371 - 0.9922
  Entity Preservation:
    Valid scores: 647/699
    Mean: 0.7532
    Range: 0.0000 - 1.0000
  BERTScore F1:
    Valid scores: 699/699
    Mean: 0.9402
    Range: 0.8732 - 0.9843

PERSONA2:
  Semantic Similarity:
    Valid scores: 699/699
    Mean: 0.9758
    Range: 0.8303 - 1.0000
  Entity Preservation:
    Valid scores: 647/699
    Mean: 0.8851
    Range: 0.0000 - 1.0000
  BERTScore F1:
    Valid scores: 699/699
    Mean: 0.9742
    Range: 0.9208 - 1.0000

PERSONA3:
  Semantic Similarity:
    Valid scores: 699/699
    Mean: 0.9331
    Range: 0.5797 - 1.0000
  Entity Preservation:
    Valid scores: 647/699
    Mean: 0.8129
    Range: 0.0000 - 1.0000
  BERTScore F1:
    Valid scores: 699/699
    Mean: 0.9503
    Range: 0.8666 - 1.0000
"""