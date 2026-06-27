"""
Script 11: Descriptive Statistics
Calculate means, medians, SDs for voice, meaning, and accessibility.
Works with updated judge_evaluations.csv containing subdimension-level LLM judge scores.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

print("=" * 70)
print("DESCRIPTIVE STATISTICS - UPDATED LLM JUDGE RESULTS")
print("=" * 70)


strategies = ["lexical", "persona1", "persona2", "persona3"]

os.makedirs("Results/Tables", exist_ok=True)
os.makedirs("Results/Figures", exist_ok=True)

# Main dimensions from new judge output
main_score_cols = [
    "voice_preservation",
    "meaning_preservation",
    "accessibility"
]

# Subdimensions from new judge output
voice_subdims = [
    "voice_professional_tone",
    "voice_interpretative_framing",
    "voice_domain_expertise",
    "voice_narrative_authority",
    "voice_rationale",
    "voice_preservation"
]

meaning_subdims = [
    "meaning_main_ideas",
    "meaning_factual_accuracy",
    "meaning_information_retention",
    "meaning_distortion_avoidance",
    "meaning_rationale",
    "meaning_preservation"
]

accessibility_subdims = [
    "accessibility_language_simplicity",
    "accessibility_terminology",
    "accessibility_sentence_clarity",
    "accessibility_audience_appropriateness",
    "accessibility_rationale",
    "accessibility"
]

all_score_cols = main_score_cols + voice_subdims + meaning_subdims + accessibility_subdims


print("\nLoading judge evaluations...")

df = pd.read_csv(
    "Data/Judge_Evaluation/judge_evaluations.csv",
    encoding="utf-8-sig"
)

print("\nColumns found:")
print(df.columns.tolist())

if "parse_success" in df.columns:
    df = df[df["parse_success"] == True].copy()

df["text_id"] = df["text_id"].astype(str).str.strip()
df["strategy"] = df["strategy"].astype(str).str.strip().str.lower()

print(f"\nLoaded {len(df)} valid judge evaluations")


missing_cols = [col for col in main_score_cols if col not in df.columns]

if missing_cols:
    raise ValueError(
        f"Missing required score columns: {missing_cols}\n"
        "Check whether your new judge_evaluations.csv was generated with the updated prompt."
    )

for col in all_score_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=main_score_cols).copy()

print(f"Rows after dropping missing main scores: {len(df)}")


print("\n" + "=" * 70)
print("MAIN DIMENSION DESCRIPTIVE STATISTICS")
print("=" * 70)

summary_data = []

for strategy in strategies:
    df_strat = df[df["strategy"] == strategy]

    if df_strat.empty:
        print(f"\n{strategy.upper()}: No data")
        continue

    voice = df_strat["voice_preservation"]
    meaning = df_strat["meaning_preservation"]
    accessibility = df_strat["accessibility"]

    print(f"\n{strategy.upper()}:")
    print(f"  N = {len(df_strat)}")
    print(f"  Voice Preservation:   M={voice.mean():.2f}, SD={voice.std():.2f}, Mdn={voice.median():.2f}")
    print(f"  Meaning Preservation: M={meaning.mean():.2f}, SD={meaning.std():.2f}, Mdn={meaning.median():.2f}")
    print(f"  Accessibility:        M={accessibility.mean():.2f}, SD={accessibility.std():.2f}, Mdn={accessibility.median():.2f}")

    summary_data.append({
        "Strategy": strategy.title(),
        "N": len(df_strat),

        "Voice_M": round(voice.mean(), 2),
        "Voice_SD": round(voice.std(), 2),
        "Voice_Mdn": round(voice.median(), 2),

        "Meaning_M": round(meaning.mean(), 2),
        "Meaning_SD": round(meaning.std(), 2),
        "Meaning_Mdn": round(meaning.median(), 2),

        "Accessibility_M": round(accessibility.mean(), 2),
        "Accessibility_SD": round(accessibility.std(), 2),
        "Accessibility_Mdn": round(accessibility.median(), 2)
    })

df_summary = pd.DataFrame(summary_data)

df_summary.to_csv(
    "Results/Tables/11_descriptive_statistics_main_dimensions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved main summary table:")
print("Results/Tables/11_descriptive_statistics_main_dimensions.csv")


print("\n" + "=" * 70)
print("SUBDIMENSION DESCRIPTIVE STATISTICS")
print("=" * 70)

subdim_records = []

available_subdims = [
    col for col in voice_subdims + meaning_subdims + accessibility_subdims
    if col in df.columns
]

for strategy in strategies:
    df_strat = df[df["strategy"] == strategy]

    for col in available_subdims:
        scores = df_strat[col].dropna()

        if scores.empty:
            continue

        if col in voice_subdims:
            group = "Voice"
        elif col in meaning_subdims:
            group = "Meaning"
        else:
            group = "Accessibility"

        subdim_records.append({
            "Strategy": strategy.title(),
            "Dimension_Group": group,
            "Subdimension": col,
            "N": len(scores),
            "Mean": round(scores.mean(), 2),
            "SD": round(scores.std(), 2),
            "Median": round(scores.median(), 2),
            "Min": round(scores.min(), 2),
            "Max": round(scores.max(), 2)
        })

df_subdims = pd.DataFrame(subdim_records)

df_subdims.to_csv(
    "Results/Tables/11_descriptive_statistics_subdimensions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Saved subdimension summary table:")
print("Results/Tables/11_descriptive_statistics_subdimensions.csv")

print("\nCreating main dimension bar plots...")

plot_summary = []

for strategy in strategies:
    df_strat = df[df["strategy"] == strategy]

    plot_summary.append({
        "Strategy": strategy.title(),
        "Voice Preservation": df_strat["voice_preservation"].mean(),
        "Meaning Preservation": df_strat["meaning_preservation"].mean(),
        "Accessibility": df_strat["accessibility"].mean()
    })

df_plot = pd.DataFrame(plot_summary)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

plot_info = [
    ("Voice Preservation", "Mean Voice Preservation"),
    ("Meaning Preservation", "Mean Meaning Preservation"),
    ("Accessibility", "Mean Accessibility")
]

for ax, (col, title) in zip(axes, plot_info):
    bars = ax.bar(
        df_plot["Strategy"],
        df_plot[col],
        edgecolor="black"
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=16)
    ax.set_ylabel("Score / 5")
    ax.set_ylim(0, 5.5)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()

        if not np.isnan(height):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.08,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

plt.tight_layout(w_pad=3)

plt.savefig(
    "Results/Figures/11_descriptive_main_dimension_barplots.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved figure:")
print("Results/Figures/11_descriptive_main_dimension_barplots.png")


print("\nCreating score distribution plot...")

dimension_cols = {
    "Voice Preservation": "voice_preservation",
    "Meaning Preservation": "meaning_preservation",
    "Accessibility": "accessibility"
}

score_order = [1, 2, 3, 4, 5]
palette = sns.color_palette("RdYlGn", n_colors=5)

fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

for ax, (title, col) in zip(axes, dimension_cols.items()):

    counts = (
        df.groupby(["strategy", col])
        .size()
        .reset_index(name="count")
    )

    pivot = counts.pivot(index="strategy", columns=col, values="count").fillna(0)
    pivot = pivot.reindex(columns=score_order, fill_value=0)
    pivot = pivot.reindex(strategies)

    row_totals = pivot.sum(axis=1)
    proportions = pivot.div(row_totals.replace(0, np.nan), axis=0) * 100
    proportions = proportions.fillna(0)

    proportions.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=palette,
        edgecolor="black",
        linewidth=0.5,
        legend=False
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Adaptation Strategy")
    ax.set_ylabel("Percentage of Texts" if ax is axes[0] else "")
    ax.set_xticklabels([s.title() for s in proportions.index], rotation=0)
    ax.set_ylim(0, 100)

handles, labels = axes[-1].get_legend_handles_labels()

if not handles:
    # Fallback: build legend handles manually from the palette
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[i]) for i in range(5)]

fig.legend(
    handles,
    [f"Score {s}" for s in score_order],
    title="Judge Score",
    bbox_to_anchor=(1.02, 0.5),
    loc="center left"
)

plt.suptitle(
    "Distribution of LLM-Judge Scores by Strategy",
    fontsize=15,
    fontweight="bold",
    y=1.03
)

plt.tight_layout()

plt.savefig(
    "Results/Figures/11_descriptive_main_dimension_boxplots.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved figure:")
print("Results/Figures/11_descriptive_main_dimension_boxplots.png")

if not df_subdims.empty:
    print("\nCreating subdimension bar plot...")

    plt.figure(figsize=(18, 8))

    sns.barplot(
        data=df_subdims,
        x="Subdimension",
        y="Mean",
        hue="Strategy"
    )

    plt.title(
        "Mean Subdimension Scores by Strategy",
        fontsize=14,
        fontweight="bold",
        pad=16
    )

    plt.ylabel("Mean Score / 5")
    plt.xlabel("Subdimension")
    plt.ylim(0, 5.5)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()

    plt.savefig(
        "Results/Figures/11_descriptive_subdimension_barplots.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved figure:")
    print("Results/Figures/11_descriptive_subdimension_barplots.png")


print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS COMPLETE")
print("=" * 70)
print("Saved outputs:")
print("  Results/Tables/11_descriptive_statistics_main_dimensions.csv")
print("  Results/Tables/11_descriptive_statistics_subdimensions.csv")
print("  Results/Figures/11_descriptive_main_dimension_barplots.png")
print("  Results/Figures/11_descriptive_main_dimension_boxplots.png")
print("  Results/Figures/11_descriptive_subdimension_barplots.png")
print("=" * 70)