"""
This script is meant to run in case there were failed evaluations
after running the LLM-as-a-judge script.
This script is an aid.
It re-runs on the same file as the one for LLM-as-a-Judge results 
and inserts the results that were failed.
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
print("RE-EVALUATING FAILED JUDGE TEXTS")
print("=" * 70)

# Setting up the API connection
load_dotenv("Config/api_keys.env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("API configured")


# Loading the same LLM-as-a-Judge prompt
with open("Prompts/llm_as_a_judge.txt", "r", encoding="utf-8") as f:
    JUDGE_PROMPT_TEMPLATE = f.read()

print("Prompt loaded")


#Loading the dataset
print("\nLoading datasets...")

# Load full dataset with adaptations
df_full = pd.read_csv(
    "Data/Merging_Files/dataset_all_adaptations.csv",
    encoding="utf-8-sig"
)

df_full = df_full.rename(columns={
    "ID": "text_id",
    "CuratorialText": "original_text",
    "lexical_adapted": "lexical",
    "persona1_adapted": "persona1",
    "persona2_adapted": "persona2",
    "persona3_adapted": "persona3"
})

# Load existing judge evaluations
df_judge = pd.read_csv(
    "Data/Judge_Evaluation/judge_evaluations.csv",
    encoding="utf-8-sig"
)

print(f"Full dataset: {len(df_full)} texts")
print(f"Judge results: {len(df_judge)} evaluations")


#This is the main loop of the code
#This identififies all the failed rowns
#As such the code only re-evaluates the failed rows, not all the ratings
print("\n" + "=" * 70)
print("IDENTIFYING FAILED EVALUATIONS")
print("=" * 70)

# Find rows where parsing failed or scores are missing
failed_mask = (
    (df_judge['parse_success'] == False) |
    (df_judge['voice_preservation'].isna()) |
    (df_judge['meaning_preservation'].isna()) |
    (df_judge['accessibility'].isna())
)

failed_rows = df_judge[failed_mask].copy()

print(f"\nTotal evaluations in file: {len(df_judge)}")
print(f"Failed evaluations: {len(failed_rows)}")

if len(failed_rows) == 0:
    print("\n✓ No failed evaluations found. All texts have valid scores.")
    print("Done.")
    exit()

print("\nFailed evaluations:")
print(failed_rows[['text_id', 'strategy', 'parse_success']].to_string())


print("\n" + "=" * 70)
print("PREPARING RETRY DATASET")
print("=" * 70)

# Get the original texts for failed evaluations
retry_tasks = []

for _, row in failed_rows.iterrows():
    text_id = row['text_id']
    strategy = row['strategy']
    
    # Find the original text
    original_row = df_full[df_full['text_id'] == text_id]
    
    if len(original_row) == 0:
        print(f"Warning: {text_id} not found in full dataset")
        continue
    
    original_text = original_row.iloc[0]['original_text']
    adapted_text = original_row.iloc[0][strategy]
    
    retry_tasks.append({
        'text_id': text_id,
        'strategy': strategy,
        'original_text': original_text,
        'adapted_text': adapted_text
    })

print(f"Tasks to retry: {len(retry_tasks)}")

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

    time.sleep(2)  # Rate limit protection
    return response.choices[0].message.content


def evaluate_one(task):
    text_id = task['text_id']
    strategy = task['strategy']
    original_text = task['original_text']
    adapted_text = task['adapted_text']

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
        print(f"\nFailed retry — {text_id} / {strategy}: {e}")

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


print("\n" + "=" * 70)
print("STARTING RE-EVALUATION")
print(f"Total calls: {len(retry_tasks)} | Workers: 2")
print("=" * 70)

MAX_WORKERS = 2

new_results = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
        executor.submit(evaluate_one, task): (task['text_id'], task['strategy'])
        for task in retry_tasks
    }

    with tqdm(total=len(futures), desc="Re-evaluating") as pbar:
        for future in as_completed(futures):
            new_results.append(future.result())
            pbar.update(1)


print("\n" + "=" * 70)
print("UPDATING RESULTS")
print("=" * 70)

# Convert new results to dataframe
new_results_df = pd.DataFrame(new_results)

# Update the original judge dataframe
for _, new_row in new_results_df.iterrows():
    mask = (
        (df_judge['text_id'] == new_row['text_id']) &
        (df_judge['strategy'] == new_row['strategy'])
    )
    
    df_judge.loc[mask, 'voice_preservation'] = new_row['voice_preservation']
    df_judge.loc[mask, 'meaning_preservation'] = new_row['meaning_preservation']
    df_judge.loc[mask, 'accessibility'] = new_row['accessibility']
    df_judge.loc[mask, 'comment'] = new_row['comment']
    df_judge.loc[mask, 'parse_success'] = new_row['parse_success']
    df_judge.loc[mask, 'raw_output'] = new_row['raw_output']

output_file = "Data/Judge_Evaluation/judge_evaluations.csv"

df_judge.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nUpdated file saved to: {output_file}")

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\nRetried: {len(new_results_df)}")
print(
    f"Now successful: {new_results_df['parse_success'].sum()} / {len(new_results_df)}"
)

print("\nRe-evaluated texts:")
print(
    new_results_df[
        [
            "text_id",
            "strategy",
            "voice_preservation",
            "meaning_preservation",
            "accessibility",
            "parse_success"
        ]
    ].to_string()
)

# Check final status
final_failed = df_judge[df_judge['parse_success'] == False]
print(f"\nRemaining failed evaluations in file: {len(final_failed)}")

print("\nFinal statistics:")
print(
    df_judge.groupby("strategy")[
        ["voice_preservation", "meaning_preservation", "accessibility"]
    ].mean(numeric_only=True).round(2)
)

print("\nDone.")