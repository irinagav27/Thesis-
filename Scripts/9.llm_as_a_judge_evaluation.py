"""
This is the script for the LLM-as-a-Judge evalaution.
"""

import pandas as pd
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import tenacity

print("=" * 70)
print("JUDGE EVALUATION - HUMAN-ALIGNED VERSION")
print("=" * 70)

load_dotenv("Config/api_keys.env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("API configured")


with open("Prompts/llm_as_a_judge.txt", "r", encoding="utf-8") as f:
    JUDGE_PROMPT_TEMPLATE = f.read()

print("Prompt loaded")

print("\nLoading dataset...")

df = pd.read_csv(
    "Data/Merging_Files/dataset_all_adaptations.csv",
    encoding="utf-8-sig"
)

df = df.rename(columns={
    "ID": "text_id",
    "CuratorialText": "original_text",
    "lexical_adapted": "lexical",
    "persona1_adapted": "persona1",
    "persona2_adapted": "persona2",
    "persona3_adapted": "persona3"
})

print(f"Loaded {len(df)} texts")

strategies = ["lexical", "persona1", "persona2", "persona3"]

print(f"\nTotal texts available: {len(df)}")
print("Enter a number, for example 5, 20, 100, or 'all'")

choice = input("\nHow many texts to evaluate? ").strip().lower()

if choice == "all":
    df_selected = df.copy()
    print(f"\nUsing ALL {len(df_selected)} texts")
else:
    n = int(choice)
    df_selected = df.head(n).copy()
    print(f"\nUsing {len(df_selected)} texts")


def parse_judge_output(judge_output):
    try:
        output = judge_output.strip()

        if "```json" in output:
            output = output.split("```json")[1].split("```")[0].strip()
        elif "```" in output:
            output = output.split("```")[1].split("```")[0].strip()

        data = json.loads(output)

        return {
            "voice_preservation": data.get("voice_preservation"),
            "meaning_preservation": data.get("meaning_preservation"),
            "accessibility": data.get("accessibility"),
            "comment": data.get("comment", ""),
            "parse_success": True
        }

    except Exception as e:
        return {
            "voice_preservation": None,
            "meaning_preservation": None,
            "accessibility": None,
            "comment": f"Parse failed: {e}",
            "parse_success": False
        }


@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=30),
    stop=tenacity.stop_after_attempt(5),
    retry=tenacity.retry_if_exception_type(Exception),
    reraise=True
)
def evaluate_text(original_text, adapted_text, strategy):
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        original=original_text,
        adapted=adapted_text,
        strategy=strategy
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        timeout=30
    )

    return response.choices[0].message.content


def evaluate_one(row, strategy):
    text_id = row["text_id"]
    original_text = row["original_text"]
    adapted_text = row[strategy]

    try:
        judge_output = evaluate_text(original_text, adapted_text, strategy)
        parsed = parse_judge_output(judge_output)

        return {
            "text_id": text_id,
            "strategy": strategy,
            "voice_preservation": parsed["voice_preservation"],
            "meaning_preservation": parsed["meaning_preservation"],
            "accessibility": parsed["accessibility"],
            "comment": parsed["comment"],
            "parse_success": parsed["parse_success"],
            "raw_output": judge_output
        }

    except Exception as e:
        print(f"\nFailed after retries — {text_id} / {strategy}: {e}")

        return {
            "text_id": text_id,
            "strategy": strategy,
            "voice_preservation": None,
            "meaning_preservation": None,
            "accessibility": None,
            "comment": f"API error: {e}",
            "parse_success": False,
            "raw_output": ""
        }


tasks = [
    (row, strategy)
    for _, row in df_selected.iterrows()
    for strategy in strategies
]

total_evaluations = len(tasks)

print("\n" + "=" * 70)
print("STARTING EVALUATION")
print(f"Total calls: {total_evaluations} | Workers: 3")
print("=" * 70)

MAX_WORKERS = 3

results = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
        executor.submit(evaluate_one, row, strategy): (row["text_id"], strategy)
        for row, strategy in tasks
    }

    with tqdm(total=len(futures), desc="Evaluating") as pbar:
        for future in as_completed(futures):
            results.append(future.result())
            pbar.update(1)


print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

os.makedirs("Data/Judge_Evaluation", exist_ok=True)

results_df = pd.DataFrame(results)

output_file = "Data/Judge_Evaluation/judge_evaluations.csv"

results_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nSaved to: {output_file}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\nTotal Evaluated: {len(results_df)}")
print(
    f"Successful: {results_df['parse_success'].sum()} "
    f"({results_df['parse_success'].mean() * 100:.1f}%)"
)
print(f"Failed: {(~results_df['parse_success']).sum()}")

print("\nSample Results:")
print(
    results_df[
        [
            "text_id",
            "strategy",
            "voice_preservation",
            "meaning_preservation",
            "accessibility"
        ]
    ].head()
)

print("\nAverage scores by strategy:")
print(
    results_df.groupby("strategy")[
        ["voice_preservation", "meaning_preservation", "accessibility"]
    ].mean(numeric_only=True).round(2)
)

print("\nOverall averages:")
print(
    results_df[
        ["voice_preservation", "meaning_preservation", "accessibility"]
    ].mean(numeric_only=True).round(2)
)

print("\nDone.")