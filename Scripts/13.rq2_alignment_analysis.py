"""
Script 13: RQ2 Analysis — FIXED VERSION
How well do LLM-judge evaluations align with linguistic metrics?
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

print("=" * 70)
print("RQ2: LLM-JUDGE vs LINGUISTIC METRICS ALIGNMENT")
print("=" * 70)


VOICE_PATH = "Data/Voice_Metrics/dataset_with_voice_metrics.csv"
SEMANTIC_PATH = "Data/Semantic_Metrics/dataset_with_semantic_metrics.csv"
JUDGE_PATH = "Data/Judge_Evaluation/judge_evaluations.csv"

TABLE_DIR = "Results/Tables"
FIGURE_DIR = "Results/Figures"

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

strategies = ["lexical", "persona1", "persona2", "persona3"]

metric_suffixes = [
    "_hedging",
    "_attribution",
    "_art_terms",
    "_semantic_sim",
    "_entity_preservation",
    "_bertscore_f1",
]

metric_labels = {
    "_hedging": "Hedging",
    "_attribution": "Attribution",
    "_art_terms": "Art Terms",
    "_semantic_sim": "Semantic Similarity",
    "_entity_preservation": "Entity Preservation",
    "_bertscore_f1": "BERTScore F1",
}

judge_dim_labels = {
    "voice_total": "Voice Preservation",
    "accuracy_total": "Meaning Preservation",
}

def standardize_columns(df):
    df = df.copy()

    rename_map = {
    "ID": "text_id",
    "id": "text_id",
    "Text ID": "text_id",
    "Text_ID": "text_id",
    "textid": "text_id",
    "text id": "text_id",

    "Strategy": "strategy",
    "adaptation_strategy": "strategy",
    "Adaptation Strategy": "strategy",

    "voice_score": "voice_total",
    "Voice Score": "voice_total",
    "voice preservation": "voice_total",
    "Voice Preservation": "voice_total",
    "voice_preservation": "voice_total",

    "accuracy_score": "accuracy_total",
    "Accuracy Score": "accuracy_total",
    "meaning_score": "accuracy_total",
    "Meaning Score": "accuracy_total",
    "meaning preservation": "accuracy_total",
    "Meaning Preservation": "accuracy_total",
    "meaning_preservation": "accuracy_total",
}
    df = df.rename(columns={col: rename_map.get(col, col) for col in df.columns})

    if "text_id" not in df.columns:
        raise ValueError("No text_id column found. Check your input file columns.")

    df["text_id"] = df["text_id"].astype(str).str.strip()

    if "strategy" in df.columns:
        df["strategy"] = (
            df["strategy"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "", regex=False)
        )

    return df


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def safe_pearson(x, y, min_n=10):
    tmp = pd.DataFrame({
        "x": safe_numeric(x),
        "y": safe_numeric(y)
    }).dropna()

    if len(tmp) < min_n:
        return np.nan, np.nan, len(tmp)

    if tmp["x"].nunique() <= 1 or tmp["y"].nunique() <= 1:
        return np.nan, np.nan, len(tmp)

    r, p = pearsonr(tmp["x"], tmp["y"])
    return r, p, len(tmp)


def wide_metrics_to_long(df_metrics, source_name):
    rows = []

    for _, row in df_metrics.iterrows():
        text_id = row["text_id"]

        for strategy in strategies:
            new_row = {
                "text_id": text_id,
                "strategy": strategy,
            }

            found_any_metric = False

            for suffix in metric_suffixes:
                original_col = f"{strategy}_adapted{suffix}"
                readable_col = metric_labels[suffix]

                if original_col in df_metrics.columns:
                    new_row[readable_col] = row[original_col]
                    found_any_metric = True

            if found_any_metric:
                rows.append(new_row)

    df_long = pd.DataFrame(rows)

    if df_long.empty:
        print(f"WARNING: No strategy-specific metric columns found in {source_name}.")
    else:
        df_long = df_long.drop_duplicates(subset=["text_id", "strategy"])

    return df_long


print("\nLoading data...")

df_voice = pd.read_csv(VOICE_PATH, encoding="utf-8-sig")
df_semantic = pd.read_csv(SEMANTIC_PATH, encoding="utf-8-sig")
df_judge = pd.read_csv(JUDGE_PATH, encoding="utf-8-sig")

df_voice = standardize_columns(df_voice)
df_semantic = standardize_columns(df_semantic)
df_judge = standardize_columns(df_judge)

print(f"Voice metrics rows: {len(df_voice)}")
print(f"Semantic metrics rows: {len(df_semantic)}")
print(f"Judge rows: {len(df_judge)}")

if "strategy" not in df_judge.columns:
    raise ValueError("judge_evaluations.csv must contain a strategy column.")

df_judge = df_judge[df_judge["strategy"].isin(strategies)].copy()

if "parse_success" in df_judge.columns:
    before = len(df_judge)
    df_judge = df_judge[
        df_judge["parse_success"].astype(str).str.lower().isin(["true", "1", "yes"])
    ].copy()
    print(f"Judge rows after parse_success filter: {len(df_judge)} / {before}")

before = len(df_judge)
df_judge = df_judge.drop_duplicates(subset=["text_id", "strategy"])

if len(df_judge) != before:
    print(f"Removed duplicate judge rows: {before - len(df_judge)}")


print("\nReshaping wide metric files into long format...")

df_voice_long = wide_metrics_to_long(df_voice, "voice metrics")
df_semantic_long = wide_metrics_to_long(df_semantic, "semantic metrics")

print(f"Voice long rows: {len(df_voice_long)}")
print(f"Semantic long rows: {len(df_semantic_long)}")

print("\nMerging on text_id + strategy...")

df_metrics = df_voice_long.merge(
    df_semantic_long,
    on=["text_id", "strategy"],
    how="outer",
    suffixes=("", "_semantic"),
)

for col in list(df_metrics.columns):
    if col.endswith("_semantic"):
        base_col = col.replace("_semantic", "")
        if base_col in df_metrics.columns:
            df_metrics[base_col] = df_metrics[base_col].combine_first(df_metrics[col])
            df_metrics = df_metrics.drop(columns=[col])

df = df_judge.merge(df_metrics, on=["text_id", "strategy"], how="inner")

print(f"Final merged rows: {len(df)}")
print(f"Unique text IDs in final data: {df['text_id'].nunique()}")
print(f"Strategies in final data: {sorted(df['strategy'].unique().tolist())}")

if df.empty:
    raise ValueError(
        "The final merged dataframe is empty. Check that text_id and strategy values match."
    )

merged_path = os.path.join(TABLE_DIR, "13_rq2_merged_analysis_dataset.csv")
df.to_csv(merged_path, index=False, encoding="utf-8-sig")


available_metrics = [label for label in metric_labels.values() if label in df.columns]
available_judge_dims = [col for col in judge_dim_labels.keys() if col in df.columns]

print("\nAvailable computational metrics:")
for col in available_metrics:
    print(f"  - {col}")

print("\nAvailable judge dimensions:")
for col in available_judge_dims:
    print(f"  - {col}: {judge_dim_labels[col]}")

if not available_metrics:
    raise ValueError("No computational metric columns found after merging.")

if not available_judge_dims:
    raise ValueError("No judge dimension/total columns found after merging.")


print("\nCalculating correlations by strategy...")

records = []

for strategy in strategies:
    df_s = df[df["strategy"] == strategy].copy()

    for judge_col in available_judge_dims:
        for metric_col in available_metrics:
            r, p, n = safe_pearson(df_s[metric_col], df_s[judge_col])

            records.append({
                "strategy": strategy,
                "judge_dimension": judge_col,
                "judge_label": judge_dim_labels[judge_col],
                "metric": metric_col,
                "pearson_r": r,
                "p_value": p,
                "n": n,
            })

df_detailed_corr = pd.DataFrame(records)

detailed_path = os.path.join(TABLE_DIR, "13_rq2_detailed_correlations_by_strategy.csv")
df_detailed_corr.to_csv(detailed_path, index=False, encoding="utf-8-sig")

print("\nBuilding average correlation matrix across strategies...")

avg_corr = (
    df_detailed_corr
    .groupby(["judge_label", "metric"], as_index=False)["pearson_r"]
    .mean()
)

df_corr_matrix = avg_corr.pivot(
    index="judge_label",
    columns="metric",
    values="pearson_r",
)

ordered_judge_labels = [judge_dim_labels[col] for col in available_judge_dims]
df_corr_matrix = df_corr_matrix.reindex(
    index=ordered_judge_labels,
    columns=available_metrics
)

matrix_path = os.path.join(TABLE_DIR, "13_rq2_average_correlation_matrix.csv")
df_corr_matrix.to_csv(matrix_path, encoding="utf-8-sig")

print("\n" + "=" * 70)
print("ALIGNMENT STRENGTH ANALYSIS")
print("=" * 70)

stacked_corr = df_corr_matrix.stack().dropna()

if len(stacked_corr) == 0:
    print("\nNo valid correlations could be calculated.")
    print("Check whether the metric columns contain variation.")
    stacked_corr = pd.Series(dtype=float)
else:
    stacked_corr = stacked_corr.sort_values(ascending=False)

print("\nSTRONG ALIGNMENT: |r| > 0.70")
strong = stacked_corr[stacked_corr.abs() > 0.70]

if strong.empty:
    print("  (None found)")
else:
    for (dim, metric), r in strong.items():
        print(f"  {metric} ↔ {dim}: r={r:.3f}")

print("\nMODERATE ALIGNMENT: 0.50 < |r| ≤ 0.70")
moderate = stacked_corr[
    (stacked_corr.abs() > 0.50) &
    (stacked_corr.abs() <= 0.70)
]

if moderate.empty:
    print("  (None found)")
else:
    for (dim, metric), r in moderate.items():
        print(f"  {metric} ↔ {dim}: r={r:.3f}")

print("\nWEAK ALIGNMENT: |r| ≤ 0.50")
weak = stacked_corr[stacked_corr.abs() <= 0.50]

if weak.empty:
    print("  (None found)")
else:
    for (dim, metric), r in weak.items():
        print(f"  {metric} ↔ {dim}: r={r:.3f}")


print("\n" + "=" * 70)
print("RANKING AGREEMENT ANALYSIS")
print("=" * 70)

ranking_records = []

ranking_metric_candidates = [
    "Semantic Similarity",
    "Entity Preservation",
    "BERTScore F1",
    "Hedging",
    "Art Terms",
]

ranking_metrics = [m for m in ranking_metric_candidates if m in df.columns]

if "voice_total" not in df.columns:
    print("voice_total not found. Ranking agreement skipped.")

elif not ranking_metrics:
    print("No suitable metrics found for ranking agreement. Skipped.")

else:
    df_rank = df[["text_id", "strategy", "voice_total"] + ranking_metrics].copy()

    for col in ["voice_total"] + ranking_metrics:
        df_rank[col] = safe_numeric(df_rank[col])

    df_rank["combined_metric_score"] = df_rank[ranking_metrics].mean(axis=1)

    for text_id, group in df_rank.groupby("text_id"):
        group = group.dropna(subset=["voice_total", "combined_metric_score"])

        if group["strategy"].nunique() < 4:
            continue

        judge_order = group.sort_values(
            "voice_total",
            ascending=False
        )["strategy"].tolist()

        metric_order = group.sort_values(
            "combined_metric_score",
            ascending=False
        )["strategy"].tolist()

        judge_ranks = {s: i + 1 for i, s in enumerate(judge_order)}
        metric_ranks = {s: i + 1 for i, s in enumerate(metric_order)}

        common_strategies = [
            s for s in strategies
            if s in judge_ranks and s in metric_ranks
        ]

        if len(common_strategies) < 4:
            continue

        judge_rank_values = [judge_ranks[s] for s in common_strategies]
        metric_rank_values = [metric_ranks[s] for s in common_strategies]

        tau, p = kendalltau(judge_rank_values, metric_rank_values)

        if not pd.isna(tau):
            ranking_records.append({
                "text_id": text_id,
                "kendall_tau": tau,
                "p_value": p,
            })

    df_ranking = pd.DataFrame(ranking_records)

    ranking_path = os.path.join(TABLE_DIR, "13_rq2_ranking_agreement.csv")
    df_ranking.to_csv(ranking_path, index=False, encoding="utf-8-sig")

    if df_ranking.empty:
        print("Insufficient complete text groups for Kendall ranking agreement.")
    else:
        mean_tau = df_ranking["kendall_tau"].mean()
        print(f"Mean Kendall's Tau: {mean_tau:.3f}")

        if mean_tau > 0.60:
            print("Interpretation: strong ranking agreement")
        elif mean_tau > 0.40:
            print("Interpretation: moderate ranking agreement")
        else:
            print("Interpretation: weak ranking agreement")


print("\nCreating heatmap...")

plt.figure(figsize=(13, max(7, len(df_corr_matrix) * 0.55)))

ax = sns.heatmap(
    df_corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "Average Pearson r across strategies"},
)

ax.set_title(
    "RQ2: Alignment Between LLM-Judge Scores and Computational Metrics\n"
    "Average Pearson Correlation Across Adaptation Strategies",
    fontsize=13,
    fontweight="bold",
    pad=16,
)

ax.set_xlabel("Computational Metric", fontsize=11)
ax.set_ylabel("Judge Dimension", fontsize=11)

plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

heatmap_path = os.path.join(FIGURE_DIR, "13_rq2_alignment_heatmap.png")
plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
plt.close()

print("\n" + "=" * 70)
print("RQ2 ANALYSIS COMPLETE")
print("=" * 70)
print(f"Merged data: {merged_path}")
print(f"Detailed correlations: {detailed_path}")
print(f"Average correlation matrix: {matrix_path}")
print(f"Heatmap: {heatmap_path}")
print("=" * 70)