"""
Script 12: RQ1 Analysis
Identify which computational metrics align with judge scores.
Updated for human-aligned judge_evaluations_full.csv:
- voice_preservation
- meaning_preservation
- accessability

"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

print("=" * 70)
print("RQ1: BEST METRICS FOR ANALYZING CURATORIAL VOICE")
print("=" * 70)


print("\nLoading data...")

df_voice = pd.read_csv(
    "Data/Voice_Metrics/dataset_with_voice_metrics.csv",
    encoding="utf-8-sig"
)

df_semantic = pd.read_csv(
    "Data/Semantic_Metrics/dataset_with_semantic_metrics.csv",
    encoding="utf-8-sig"
)

df_judge = pd.read_csv(
    "Data/Judge_Evaluation/judge_evaluations.csv",
    encoding="utf-8-sig"
)

print("\nVoice metrics columns:")
print(df_voice.columns.tolist())

print("\nSemantic metrics columns:")
print(df_semantic.columns.tolist())

print("\nJudge evaluation columns:")
print(df_judge.columns.tolist())


def standardize_id_column(dataframe):
    dataframe = dataframe.rename(columns={
        "ID": "text_id",
        "id": "text_id",
        "Text ID": "text_id",
        "textid": "text_id",
        "Text_ID": "text_id"
    })

    if "text_id" not in dataframe.columns:
        raise ValueError("No ID/text_id column found in one of the files.")

    dataframe["text_id"] = dataframe["text_id"].astype(str).str.strip()
    return dataframe


df_voice = standardize_id_column(df_voice)
df_semantic = standardize_id_column(df_semantic)
df_judge = standardize_id_column(df_judge)


print("\nCleaning judge evaluations...")

df_judge["strategy"] = df_judge["strategy"].astype(str).str.strip().str.lower()

df_judge["parse_success"] = (
    df_judge["parse_success"]
    .astype(str)
    .str.lower()
    .isin(["true", "1", "yes"])
)

df_judge = df_judge[df_judge["parse_success"] == True].copy()

df_judge["voice_preservation"] = pd.to_numeric(
    df_judge["voice_preservation"],
    errors="coerce"
)

df_judge["meaning_preservation"] = pd.to_numeric(
    df_judge["meaning_preservation"],
    errors="coerce"
)

print(f"Valid judge rows: {len(df_judge)}")

print("\nJudge strategy counts:")
print(df_judge["strategy"].value_counts())

print("\nVoice preservation score distribution:")
print(df_judge["voice_preservation"].value_counts(dropna=False).sort_index())

print("\nMeaning preservation score distribution:")
print(df_judge["meaning_preservation"].value_counts(dropna=False).sort_index())


print("\nMerging computational metric files...")

df_metrics = df_voice.merge(
    df_semantic,
    on="text_id",
    how="left",
    suffixes=("", "_semantic")
)

print(f"Metric rows: {len(df_metrics)}")

strategies = ["lexical", "persona1", "persona2", "persona3"]

os.makedirs("Results/Tables", exist_ok=True)
os.makedirs("Results/Figures", exist_ok=True)

# metric_pairs: (column suffix, judge dimension, metric label, judge label)
# Entity Preservation is now included as a third semantic metric.
metric_pairs = [
    ("_hedging", "voice_preservation", "Hedging Density", "Voice Preservation"),
    ("_attribution", "voice_preservation", "Attribution Markers", "Voice Preservation"),
    ("_art_terms", "voice_preservation", "Art Term Density", "Voice Preservation"),
    ("_semantic_sim", "meaning_preservation", "Semantic Similarity", "Meaning Preservation"),
    ("_bertscore_f1", "meaning_preservation", "BERTScore F1", "Meaning Preservation"),
    ("_entity_preservation", "meaning_preservation", "Entity Preservation", "Meaning Preservation"),
]


print("\nChecking required columns...\n")

for metric_suffix, judge_col, metric_name, judge_name in metric_pairs:
    for strategy in strategies:
        metric_col = f"{strategy}_adapted{metric_suffix}"

        if metric_col not in df_metrics.columns:
            print(f"Missing metric column: {metric_col}")

    if judge_col not in df_judge.columns:
        print(f"Missing judge column: {judge_col}")


def calculate_strategy_correlation(strategy, metric_col, judge_col, metric_name):
    df_metric = df_metrics[["text_id", metric_col]].copy()

    df_judge_strategy = df_judge[
        df_judge["strategy"] == strategy
    ][["text_id", judge_col]].copy()

    df_valid = df_metric.merge(
        df_judge_strategy,
        on="text_id",
        how="inner"
    )

    df_valid[metric_col] = pd.to_numeric(df_valid[metric_col], errors="coerce")
    df_valid[judge_col] = pd.to_numeric(df_valid[judge_col], errors="coerce")

    df_valid = df_valid.dropna()

    print(f"\nDEBUG {strategy.upper()} - {metric_name}")
    print(f"  Metric column: {metric_col}")
    print(f"  Judge column: {judge_col}")
    print(f"  Rows after merge/dropna: {len(df_valid)}")
    print(f"  Metric unique values: {df_valid[metric_col].nunique()}")
    print(f"  Judge unique values: {df_valid[judge_col].nunique()}")

    if len(df_valid) > 0:
        print("  First rows:")
        print(df_valid[[metric_col, judge_col]].head().to_string(index=False))

    if len(df_valid) < 10:
        print(f"  {strategy}: Insufficient data")
        return np.nan, np.nan, len(df_valid)

    if df_valid[metric_col].nunique() <= 1:
        print(f"  {strategy}: Metric has no variation")
        return np.nan, np.nan, len(df_valid)

    if df_valid[judge_col].nunique() <= 1:
        print(f"  {strategy}: Judge score has no variation")
        return np.nan, np.nan, len(df_valid)

    r, p = pearsonr(df_valid[metric_col], df_valid[judge_col])

    print(f"  {strategy}: r={r:.3f}, p={p:.4f}")

    return r, p, len(df_valid)


print("\nCorrelating computational metrics with judge scores...\n")

correlation_results = []
heatmap_rows = []

for metric_suffix, judge_col, metric_name, judge_name in metric_pairs:
    print("\n" + "-" * 70)
    print(f"{metric_name} <-> {judge_name}")
    print("-" * 70)

    correlations_by_strategy = []
    heatmap_row = []

    for strategy in strategies:
        metric_col = f"{strategy}_adapted{metric_suffix}"

        if metric_col not in df_metrics.columns:
            print(f"{strategy}: Missing metric column {metric_col}")
            heatmap_row.append(np.nan)
            continue

        if judge_col not in df_judge.columns:
            print(f"{strategy}: Missing judge column {judge_col}")
            heatmap_row.append(np.nan)
            continue

        r, p, n = calculate_strategy_correlation(
            strategy=strategy,
            metric_col=metric_col,
            judge_col=judge_col,
            metric_name=metric_name
        )

        heatmap_row.append(r)

        if not pd.isna(r):
            correlations_by_strategy.append(r)

    avg_r = np.mean(correlations_by_strategy) if correlations_by_strategy else np.nan

    if pd.isna(avg_r):
        print("\nAverage: NaN")
    else:
        print(f"\nAverage: r={avg_r:.3f}")

    correlation_results.append({
        "Metric": metric_name,
        "Judge Dimension": judge_name,
        "Avg Correlation": avg_r,
        "Strong (r>0.6)": "Yes" if not pd.isna(avg_r) and abs(avg_r) > 0.6 else "No",
        "Good (r>0.5)": "Yes" if not pd.isna(avg_r) and abs(avg_r) > 0.5 else "No"
    })

    heatmap_rows.append(heatmap_row)

df_corr = pd.DataFrame(correlation_results)

df_corr.to_csv(
    "Results/Tables/12_rq1_correlations.csv",
    index=False,
    encoding="utf-8-sig"
)


print("\n" + "=" * 70)
print("BEST METRICS IDENTIFIED:")
print("=" * 70)

df_corr_sorted = df_corr.sort_values(
    "Avg Correlation",
    ascending=False,
    na_position="last",
    key=lambda s: s.abs()
)

for _, row in df_corr_sorted.iterrows():
    print(f"\n{row['Metric']} -> {row['Judge Dimension']}")

    if pd.isna(row["Avg Correlation"]):
        print("  Avg Correlation: NaN")
        print("  Not enough valid data or no score variation")
    else:
        print(f"  Avg Correlation: {row['Avg Correlation']:.3f}")

        if row["Strong (r>0.6)"] == "Yes":
            print("  EXCELLENT alignment")
        elif row["Good (r>0.5)"] == "Yes":
            print("  GOOD alignment")
        else:
            print("  Weak or limited alignment")


print("\nCreating correlation heatmap...")

df_heatmap = pd.DataFrame(
    heatmap_rows,
    index=[m[2] for m in metric_pairs],
    columns=[s.title() for s in strategies]
)

df_heatmap = df_heatmap.dropna(how="all")

df_heatmap.to_csv(
    "Results/Tables/12_rq1_heatmap_values.csv",
    encoding="utf-8-sig"
)

if df_heatmap.empty:
    print("Heatmap skipped because all correlation values are NaN.")
else:
    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        df_heatmap,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Pearson r"},
        linewidths=0.5,
        linecolor="white"
    )

    ax.set_title(
        "RQ1: Computational Metrics vs Judge Scores\n"
        "Correlation by Strategy",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xlabel("Strategy", fontsize=11)
    ax.set_ylabel("Computational Metric", fontsize=11)

    plt.tight_layout()

    plt.savefig(
        "Results/Figures/12_rq1_correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Figure saved: Results/Figures/12_rq1_correlation_heatmap.png")


print("\n" + "=" * 70)
print("RQ1 ANALYSIS COMPLETE")
print("=" * 70)
print("Results saved to: Results/Tables/12_rq1_correlations.csv")
print("Heatmap values saved to: Results/Tables/12_rq1_heatmap_values.csv")
print("Visualization saved to: Results/Figures/12_rq1_correlation_heatmap.png")
print("=" * 70)