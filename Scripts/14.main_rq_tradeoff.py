"""
Script 14: Main RQ Analysis
Quantify the trade-off between accessibility and voice/meaning preservation.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

print("=" * 70)
print("MAIN RQ: ACCESSIBILITY-PRESERVATION TRADE-OFF")
print("=" * 70)


print("\nLoading judge evaluations...")

df_judge = pd.read_csv(
    'Data/Judge_Evaluation/judge_evaluations.csv',
    encoding='utf-8-sig'
)

df_judge = df_judge[df_judge['parse_success'] == True].copy()

df_judge['text_id'] = df_judge['text_id'].astype(str).str.strip()
df_judge['strategy'] = df_judge['strategy'].astype(str).str.strip().str.lower()

print(f"Loaded {len(df_judge)} valid judge evaluations")


print("\nLoading human accessibility scores...")

human_eval_files = {
    'lexical': 'Data/Human_Evaluation/human_evaluation_lexical_699texts_20260606_122318.xlsx',
    'persona1': 'Data/Human_Evaluation/human_evaluation_persona1_699texts_20260606_122352.xlsx',
    'persona2': 'Data/Human_Evaluation/human_evaluation_persona2_699texts_20260606_122420.xlsx',
    'persona3': 'Data/Human_Evaluation/human_evaluation_persona3_699texts_20260606_122432.xlsx',
}

accessibility_data = []

for strategy, path in human_eval_files.items():
    if not os.path.exists(path):
        print(f"  Missing file for {strategy}: {path}")
        continue

    df_human = pd.read_excel(path)

    accessibility_cols = [c for c in df_human.columns if 'accessibility' in c.lower()]
    id_cols = [c for c in df_human.columns if 'id' in c.lower()]

    if not accessibility_cols or not id_cols:
        print(f"  Could not find accessibility/id column for {strategy}")
        print(f"  Columns: {df_human.columns.tolist()}")
        continue

    accessibility_col = accessibility_cols[0]
    id_col = id_cols[0]

    count = 0
    for _, row in df_human.iterrows():
        score = row[accessibility_col]
        if pd.isna(score):
            continue
        accessibility_data.append({
            'text_id': str(row[id_col]).strip(),
            'strategy': strategy,
            'accessibility_score': float(score)
        })
        count += 1

    print(f"  Loaded {count} accessibility scores for {strategy}")

df_accessibility = pd.DataFrame(accessibility_data)


print("\nMerging judge and accessibility data...")

df_judge = df_judge.merge(
    df_accessibility,
    on=['text_id', 'strategy'],
    how='left'
)

df_judge = df_judge.dropna(
    subset=['voice_preservation', 'meaning_preservation', 'accessibility_score']
).copy()

print(f"Final merged dataset: {len(df_judge)} rows\n")


strategies = ['lexical', 'persona1', 'persona2', 'persona3']

os.makedirs('Results/Tables', exist_ok=True)
os.makedirs('Results/Figures', exist_ok=True)

# All judge/human scores are on a 1-5 scale, so percentages use /5.
SCALE_MAX = 5


print("=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

r_voice, p_voice = pearsonr(
    df_judge['voice_preservation'],
    df_judge['accessibility_score']
)

r_meaning, p_meaning = pearsonr(
    df_judge['meaning_preservation'],
    df_judge['accessibility_score']
)

print(f"\nVoice <-> Accessibility: r={r_voice:.3f}, p={p_voice:.4f}")
if r_voice < -0.4:
    print("  Moderate-to-strong trade-off exists")
elif r_voice < -0.2:
    print("  Weak trade-off exists")
else:
    print("  No strong trade-off detected")

print(f"\nMeaning <-> Accessibility: r={r_meaning:.3f}, p={p_meaning:.4f}")
if abs(r_meaning) < 0.2:
    print("  Meaning can be preserved largely independently of accessibility")
else:
    print("  Meaning and accessibility show some relationship")


print("\n" + "=" * 70)
print("STRATEGY EFFICIENCY ANALYSIS")
print("=" * 70)

efficiency_results = []

for strategy in strategies:
    df_strat = df_judge[df_judge['strategy'] == strategy]

    if len(df_strat) == 0:
        print(f"\n{strategy.upper()}: No data")
        continue

    # CORRECTED: all three divide by 5 (their actual scale), not /25 or /20.
    voice_pct = (df_strat['voice_preservation'].mean() / SCALE_MAX) * 100
    meaning_pct = (df_strat['meaning_preservation'].mean() / SCALE_MAX) * 100
    accessibility_pct = (df_strat['accessibility_score'].mean() / SCALE_MAX) * 100

    voice_loss = 100 - voice_pct
    meaning_loss = 100 - meaning_pct

    if (voice_loss + meaning_loss) > 0:
        efficiency = accessibility_pct / (voice_loss + meaning_loss)
    else:
        efficiency = np.inf

    print(f"\n{strategy.upper()}:")
    print(f"  Voice: {voice_pct:.1f}% preserved")
    print(f"  Meaning: {meaning_pct:.1f}% preserved")
    print(f"  Accessibility: {accessibility_pct:.1f}%")
    print(f"  Efficiency Ratio: {efficiency:.2f}")

    efficiency_results.append({
        'Strategy': strategy.title(),
        'Voice %': f"{voice_pct:.1f}%",
        'Meaning %': f"{meaning_pct:.1f}%",
        'Accessibility %': f"{accessibility_pct:.1f}%",
        'Efficiency': efficiency
    })

df_efficiency = pd.DataFrame(efficiency_results)
df_efficiency.to_csv('Results/Tables/14_strategy_efficiency.csv', index=False)


print("\n" + "=" * 70)
print("EFFICIENCY RANKING")
print("=" * 70)

df_eff_sorted = df_efficiency.sort_values('Efficiency', ascending=False)

for _, row in df_eff_sorted.iterrows():
    print(f"{row['Strategy']}: {row['Efficiency']:.2f}")


print("\nCreating trade-off bar plot...")

tradeoff_summary = []

for strategy in strategies:
    df_strat = df_judge[df_judge['strategy'] == strategy]

    tradeoff_summary.append({
        'Strategy': strategy.title(),
        'Voice Preservation (%)': (df_strat['voice_preservation'].mean() / SCALE_MAX) * 100,
        'Meaning Preservation (%)': (df_strat['meaning_preservation'].mean() / SCALE_MAX) * 100,
        'Accessibility (%)': (df_strat['accessibility_score'].mean() / SCALE_MAX) * 100
    })

df_tradeoff = pd.DataFrame(tradeoff_summary)

x = np.arange(len(df_tradeoff['Strategy']))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - width, df_tradeoff['Voice Preservation (%)'], width, label='Voice Preservation')
bars2 = ax.bar(x, df_tradeoff['Meaning Preservation (%)'], width, label='Meaning Preservation')
bars3 = ax.bar(x + width, df_tradeoff['Accessibility (%)'], width, label='Accessibility')

ax.set_title('Main RQ: Accessibility vs Preservation by Strategy', fontsize=13, fontweight='bold')
ax.set_ylabel('Mean Score (%)')
ax.set_xticks(x)
ax.set_xticklabels(df_tradeoff['Strategy'])
ax.set_ylim(0, 110)
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('Results/Figures/14_tradeoff_barplot.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure saved: Results/Figures/14_tradeoff_barplot.png")


print("\n" + "=" * 70)
print("ACCEPTABILITY THRESHOLD ANALYSIS")
print("=" * 70)

thresholds = {
    'Voice': 70,
    'Meaning': 85,
    'Accessibility': 40
}

threshold_results = []

for strategy in strategies:
    df_strat = df_judge[df_judge['strategy'] == strategy]

    if len(df_strat) == 0:
        continue

    voice_pct = (df_strat['voice_preservation'].mean() / SCALE_MAX) * 100
    meaning_pct = (df_strat['meaning_preservation'].mean() / SCALE_MAX) * 100
    access_pct = (df_strat['accessibility_score'].mean() / SCALE_MAX) * 100

    voice_ok = voice_pct >= thresholds['Voice']
    meaning_ok = meaning_pct >= thresholds['Meaning']
    access_ok = access_pct >= thresholds['Accessibility']
    meets_all = voice_ok and meaning_ok and access_ok

    print(f"\n{strategy.title()}:")
    print(f"  Voice: {voice_pct:.1f}% {'PASS' if voice_ok else 'FAIL'}")
    print(f"  Meaning: {meaning_pct:.1f}% {'PASS' if meaning_ok else 'FAIL'}")
    print(f"  Accessibility: {access_pct:.1f}% {'PASS' if access_ok else 'FAIL'}")
    print(f"  Overall: {'MEETS ALL THRESHOLDS' if meets_all else 'Does not meet all thresholds'}")

    threshold_results.append({
        'Strategy': strategy.title(),
        'Voice >=70%': '+' if voice_ok else '-',
        'Meaning >=85%': '+' if meaning_ok else '-',
        'Accessibility >=40%': '+' if access_ok else '-',
        'Meets All': '+' if meets_all else '-'
    })

df_thresholds = pd.DataFrame(threshold_results)
df_thresholds.to_csv('Results/Tables/14_acceptability_thresholds.csv', index=False)

print("\n" + "=" * 70)
print("MAIN RQ ANALYSIS COMPLETE")
print("=" * 70)
print("\nSaved outputs:")
print("  Results/Tables/14_strategy_efficiency.csv")
print("  Results/Tables/14_acceptability_thresholds.csv")
print("  Results/Figures/14_tradeoff_barplot.png")