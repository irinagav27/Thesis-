import glob, os
import pandas as pd

folder = "Data/Human_Evaluation"
DIMS = ["Voice Preservation (1-5)", "Meaning Preservation (1-5)", "Accessibility (1-5)"]  # use your real names

STRATS = [("Lexical", "lexical"), ("Persona 1", "persona1"),
          ("Persona 2", "persona2"), ("Persona 3", "persona3")]

rows = []
for label, key in STRATS:
    matches = glob.glob(os.path.join(folder, f"human_evaluation_{key}_*.xlsx"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{key}: expected 1 file, found {len(matches)}: {matches}")
    path = matches[0]

    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()  # clean stray header spaces

    for col in DIMS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_bad = df[DIMS].isna().sum().sum()
    print(f"{label}: {len(df)} rows, {n_bad} non-numeric cells skipped  ({os.path.basename(path)})")

    means = df[DIMS].mean()
    rows.append({
        "strategy": label,
        "voice_preservation":   round(means[DIMS[0]], 2),
        "meaning_preservation": round(means[DIMS[1]], 2),
        "accessibility":        round(means[DIMS[2]], 2),
    })

human_means = pd.DataFrame(rows).set_index("strategy")
print("\nHuman means per strategy:")
print(human_means)

human_means.to_csv("Data/Human_Evaluation/human_strategy_means.csv", encoding="utf-8-sig")