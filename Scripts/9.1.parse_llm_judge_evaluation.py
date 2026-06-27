"""
Script: Regenerate judge_evaluations.csv with full dimension and
subdimension columns.

This script parses 'raw_output' for every row, extracts every field
the judge prompt asked for, and writes a new CSV with one column per
dimension/subdimension. 
"""

import pandas as pd
import json
import os
import warnings

warnings.filterwarnings('ignore')

print("=" * 70)
print("REGENERATING judge_evaluations.csv WITH FULL SUBDIMENSIONS")
print("=" * 70)

# ============================================================
# SETTINGS
# ============================================================

INPUT_PATH = "Data/Judge_Evaluation/judge_evaluations.csv"
OUTPUT_PATH = "Data/Judge_Evaluation/judge_evaluations_parsed.csv"

# Every field the LLM-as-a-judge prompt asks for, in the order
# it should appear in the output CSV.
EXPECTED_FIELDS = [
    # Voice preservation
    "voice_professional_tone",
    "voice_interpretative_framing",
    "voice_domain_expertise",
    "voice_narrative_authority",
    "voice_rationale",
    "voice_preservation",

    # Meaning preservation
    "meaning_main_ideas",
    "meaning_factual_accuracy",
    "meaning_information_retention",
    "meaning_distortion_avoidance",
    "meaning_rationale",
    "meaning_preservation",

    # Accessibility
    "accessibility_language_simplicity",
    "accessibility_terminology",
    "accessibility_sentence_clarity",
    "accessibility_audience_appropriateness",
    "accessibility_rationale",
    "accessibility",
]

# Numeric fields - everything except the three rationale text fields
NUMERIC_FIELDS = [f for f in EXPECTED_FIELDS if not f.endswith("_rationale")]

# ============================================================
# LOAD EXISTING FILE
# ============================================================

print(f"\nLoading {INPUT_PATH} ...")

df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

print(f"Loaded {len(df)} rows")
print(f"Existing columns: {df.columns.tolist()}")

if "raw_output" not in df.columns:
    raise ValueError(
        "No 'raw_output' column found in the input CSV. "
        "Cannot recover subdimension data without it."
    )

# ============================================================
# PARSE raw_output JSON FOR EVERY ROW
# ============================================================

print("\nParsing raw_output JSON for every row...")

parsed_rows = []
parse_errors = []

for idx, row in df.iterrows():
    raw = row.get("raw_output")

    record = {field: None for field in EXPECTED_FIELDS}

    if pd.isna(raw) or str(raw).strip() == "":
        parse_errors.append((idx, row.get("text_id"), row.get("strategy"), "empty raw_output"))
        parsed_rows.append(record)
        continue

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        parse_errors.append((idx, row.get("text_id"), row.get("strategy"), str(e)))
        parsed_rows.append(record)
        continue

    for field in EXPECTED_FIELDS:
        record[field] = parsed.get(field, None)

    parsed_rows.append(record)

df_parsed = pd.DataFrame(parsed_rows)

print(f"Parsed {len(df_parsed)} rows")
print(f"Rows with parse errors: {len(parse_errors)}")

if parse_errors:
    print("\nFirst few parse errors:")
    for idx, text_id, strategy, err in parse_errors[:5]:
        print(f"  Row {idx} ({text_id}, {strategy}): {err}")

# ============================================================
# CONVERT NUMERIC FIELDS
# ============================================================

for field in NUMERIC_FIELDS:
    df_parsed[field] = pd.to_numeric(df_parsed[field], errors="coerce")

# ============================================================
# COMBINE WITH ORIGINAL METADATA COLUMNS
# ============================================================

meta_cols = ["text_id", "strategy", "comment", "parse_success"]
meta_cols = [c for c in meta_cols if c in df.columns]

df_final = pd.concat(
    [df[meta_cols].reset_index(drop=True), df_parsed.reset_index(drop=True)],
    axis=1
)

# Keep raw_output at the very end, for traceability / re-parsing if needed
if "raw_output" in df.columns:
    df_final["raw_output"] = df["raw_output"].reset_index(drop=True)

# ============================================================
# SANITY CHECKS
# ============================================================

print("\n" + "=" * 70)
print("SANITY CHECKS")
print("=" * 70)

print(f"\nFinal shape: {df_final.shape}")
print(f"Final columns: {df_final.columns.tolist()}")

print("\nMissing values per subdimension column:")
for field in EXPECTED_FIELDS:
    n_missing = df_final[field].isna().sum()
    if n_missing > 0:
        print(f"  {field}: {n_missing} missing")

print("\nVoice subdimension columns found:")
voice_cols = [c for c in df_final.columns if c.startswith("voice_") and c != "voice_rationale"]
print(f"  {voice_cols}")

print("\nMeaning subdimension columns found:")
meaning_cols = [c for c in df_final.columns if c.startswith("meaning_") and c != "meaning_rationale"]
print(f"  {meaning_cols}")

print("\nAccessibility subdimension columns found:")
accessibility_cols = [
    c for c in df_final.columns
    if c.startswith("accessibility_") and c != "accessibility_rationale"
]
print(f"  {accessibility_cols}")

# Quick spot-check: row count per strategy should match original
print("\nRow count per strategy (should match original):")
print(df_final["strategy"].value_counts())

# ============================================================
# SAVE
# ============================================================

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

df_final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print(f"Saved full dataset to: {OUTPUT_PATH}")
print(
    "\nPoint downstream scripts (e.g. Script 15, Cronbach's alpha) "
    "at this new file instead of the original judge_evaluations.csv."
)
print("=" * 70)

# ============================================================
# OVERALL SCORES PER STRATEGY 
# ============================================================

print("\n" + "=" * 70)
print("MEAN SCORES PER STRATEGY")
print("=" * 70)

# Mean of every numeric judge field, grouped by strategy
strategy_means = df_final.groupby("strategy")[NUMERIC_FIELDS].mean()

# The three aggregate dimensions are what Table 6 reports
aggregate_cols = ["voice_preservation", "meaning_preservation", "accessibility"]

print("\nAggregate dimensions (the three Table 6 columns):")
print(strategy_means[aggregate_cols].round(2))

print("\nAll dimensions + subdimensions:")
print(strategy_means.round(2).to_string())

# Save both so you can paste straight into the paper
SUMMARY_PATH = "Data/Judge_Evaluation/judge_strategy_means.csv"
strategy_means.round(4).to_csv(SUMMARY_PATH, encoding="utf-8-sig")
print(f"\nSaved per-strategy means to: {SUMMARY_PATH}")

