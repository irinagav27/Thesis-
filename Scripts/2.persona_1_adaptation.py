"""
This is the file for the generation of the Persona 1 Adaptation.
The profile of this persona is the following:
* Age: around 13-15 years old
* Reading comprehension: 8th grader (Flesch-Kincaid ~6-8)
* Art knowledge: Minimal; it entails school-level exposure only
* English proficiency: Native English Speaker
What case will this test:
* Combined lexical and syntactic simplification
* Impact on curatorial voice when targeting young readers
* Balance between accessibility and content preservation
* Maximum readability improvement scenario
"""

import pandas as pd
from openai import OpenAI
from httpx import Timeout
from dotenv import load_dotenv
import os
import time
from tqdm import tqdm

#Loading the API key
load_dotenv('Config/api_keys.env')

"""
This part of the code creates a client to connect to the Deepseek API.
A timeout of 120 seconds is set to handle a longer processing time, and if the model takes longer
than 120 second to respond, the request fails.
"""
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    timeout=Timeout(120.0)
)


#Function to retry in case of errors
def call_deepseek_with_retry(messages, max_retries=8):
    """this function is a helper function, which has the purpose of calling the
    Deepseek API, and retry automatically if it fails. It can fail up to 8 times """
    for attempt in range(max_retries):

        try:

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.3,
                max_tokens=1200
            )
            """This is meant to call to the API, and get a responde. 
            In this case a lower temperature is used
            to make the output more deterministic, more stable and less creative.
            As there is no creativiy needed for this task, and we want the output to
            be as consistent as possible"""
            return response

        except Exception as e:
            # This is meant to handle the error(such as timeout, API failure, connection issues)
            wait_time = min(5 * (2 ** attempt), 60)  # calculates waiting time before trying again
            #The wait_time is called an exponential backoff, meaning each waiting time increases exponentially 
            #With each retry up to 60 seconds
            print(f"\nRetry {attempt + 1}/{max_retries}")
            print(f"Error: {e}")
            print(f"Waiting {wait_time} seconds...\n")

            time.sleep(wait_time)

    raise Exception("Max retries exceeded")



#Loading the prompt
print("Loading Persona 1 prompt...")

with open('Prompts/persona1_prompt.txt', 'r', encoding='utf-8-sig') as f: #This is meant to load the prompt for the adaptation based on the profile of persona 1.
    PROMPT_TEMPLATE = f.read()
#It was chosen to have a different file for the prompts, as such if there was a need to change the prompt
#Not the entire code needed changes

#Loading the dataset
print("Loading dataset...")

df = pd.read_excel('Data/Original/Dataset_Curatorial_Text.xlsx') #Loading the dataset of curatorial texts

print(f"Found {len(df)} texts to process\n")

#The output paths, if the folder does not exist, one will be created
os.makedirs('Data/adapted/Persona_1', exist_ok=True)
""""
There will be 4 files for the output:
*checkpoint_path: This file is meant to monitor the intermediate results, to monitor the results
*failed_output_path: This file is meant to save all the failed rows
*final_output_path: This file is meant to save the final adapted texts, based on the profile of persona 1
*failed_ids_path: This file is meant to save the IDs of all the failed rows
"""
checkpoint_path = 'Data/adapted/Persona_1/persona1_checkpoint.csv'
final_output_path = 'Data/adapted/Persona_1/persona1_adapted.csv'
failed_output_path = 'Data/adapted/Persona_1/persona1_failed_rows.csv'
failed_ids_path = 'Data/adapted/Persona_1/persona1_failed_ids.txt'


#Loading the checkpoint, if one exists
processed_ids = set()

if os.path.exists(checkpoint_path):

    checkpoint_df = pd.read_csv(checkpoint_path)

    results = checkpoint_df.to_dict('records')

    processed_ids = set(checkpoint_df['ID'])

    print(f"Loaded checkpoint with {len(processed_ids)} completed texts\n")

else:

    results = []

"""
This is the main processing loop of the code.
For each curatorial text, this loop will create an output of the persona 1 version
of the initial text, and it will also keep track of the word counts of the original and
of the adapted texts.

"""

for idx, row in tqdm(df.iterrows(),
                     total=len(df),
                     desc="Persona 1 Adaptation"):

    text_id = row['ID']

    # Skip already processed rows
    if text_id in processed_ids:
        continue

    original_text = row['CuratorialText']

    #This section is for handling empty texts
    if pd.isna(original_text) or str(original_text).strip() == "":

        print(f"Skipping {text_id} - empty text")

        results.append({
            'ID': text_id,
            'CuratorialText': original_text,
            'persona1_adapted': None,
            'word_count_original': 0,
            'word_count_adapted': 0,
            'error': 'Empty original text'
        })

        continue

  
    full_prompt = f"""
{PROMPT_TEMPLATE}

Original text:
\"\"\"
{original_text}
\"\"\"

Provide the adapted text for an 8th-grade reader:
"""



    try:

        messages = [
            {
                "role": "system",
                "content": "You are an expert at adapting museum texts for young readers."
            },
            {
                "role": "user",
                "content": full_prompt
            }
        ]

        response = call_deepseek_with_retry(messages)

        adapted_text = response.choices[0].message.content.strip()



        results.append({
            'ID': text_id, # the identifier made to check the number of the curatorial text
            'CuratorialText': original_text, # The original curatorial text from the museum websites
            'persona1_adapted': adapted_text, #Text after processed
            'word_count_original': len(str(original_text).split()),
            'word_count_adapted': len(adapted_text.split()),
            'error': None
        })



        if len(results) % 5 == 0: #Every 5 steps it saved a checkpoint

            pd.DataFrame(results).to_csv(
                checkpoint_path,
                index=False,
                encoding='utf-8-sig'
            )

            print(f" Checkpoint saved ({len(results)} rows)")


        time.sleep(1)

# In case there is an error
#This is made as such if there is one error row, not the entire code will fail
#And only one row will be lost.
    except Exception as e:

        print(f"\n Error processing {text_id}: {e}")

        results.append({
            'ID': text_id,
            'CuratorialText': original_text,
            'persona1_adapted': None,
            'word_count_original': len(str(original_text).split()),
            'word_count_adapted': 0,
            'error': str(e)
        })

        # Save checkpoint after errors too
        pd.DataFrame(results).to_csv(
            checkpoint_path,
            index=False,
            encoding='utf-8-sig'
        )

        time.sleep(3)


# A second pass in case of an error
print("\nStarting second pass for failed rows...")

failed_entries = [r for r in results if r['error'] is not None]

if len(failed_entries) > 0:

    recovered = []
    still_failed = []

    print(f"Retrying {len(failed_entries)} failed texts...\n")

    for entry in tqdm(failed_entries, desc="Retry Failed Rows"):

        text_id = entry['ID']
        original_text = entry['CuratorialText']

        try:

            full_prompt = f"""
{PROMPT_TEMPLATE}

Original text:
\"\"\"
{original_text}
\"\"\"

Provide the adapted text for an 8th-grade reader:
"""

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at adapting museum texts for young readers."
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]

            response = call_deepseek_with_retry(messages)

            adapted_text = response.choices[0].message.content.strip()

            recovered.append({
                'ID': text_id,
                'CuratorialText': original_text,
                'persona1_adapted': adapted_text,
                'word_count_original': len(str(original_text).split()),
                'word_count_adapted': len(adapted_text.split()),
                'error': None
            })

            print(f" Recovered {text_id}")

            time.sleep(1)

        except Exception as e:

            still_failed.append({
                'ID': text_id,
                'CuratorialText': original_text,
                'persona1_adapted': None,
                'word_count_original': len(str(original_text).split()),
                'word_count_adapted': 0,
                'error': str(e)
            })

            print(f" Still failed: {text_id}")

    # Keep successful rows
    successful_rows = [r for r in results if r['error'] is None]

    results = successful_rows + recovered + still_failed



print("\nSaving final results...")

df_results = pd.DataFrame(results)

df_results.to_csv(
    final_output_path,
    index=False,
    encoding='utf-8-sig'
)


#Saving failed rows
failed_rows = df_results[df_results['error'].notna()]

if len(failed_rows) > 0:

    failed_rows.to_csv(
        failed_output_path,
        index=False,
        encoding='utf-8-sig' 
    )

    # Save failed IDs separately
    with open(failed_ids_path, 'w', encoding='utf-8-sig') as f:

        for text_id in failed_rows['ID']:
            f.write(f"{text_id}\n")

    print("\nFailed IDs:")
    print(failed_rows['ID'].tolist())


#Summary afterwards
successful = df_results['error'].isna().sum()
failed = len(df_results) - successful

print("\n" + "=" * 60)
print("PERSONA 1 ADAPTATION COMPLETE")
print("=" * 60)
print(f" Successful: {successful}/{len(df_results)}")
print(f" Failed: {failed}/{len(df_results)}")
print(f" Final results: {final_output_path}")
print(f" Checkpoint file: {checkpoint_path}")

if failed > 0:
    print(f" Failed rows: {failed_output_path}")
    print(f" Failed IDs: {failed_ids_path}")

print("=" * 60)
""""
1: After the first run we have 670/699 texts adapted.
There were 29 failures.
It is to be noticed, that for the failures, there is not even the wordcount of the initial texts present either. But there is a row for Requested time out.
Possible amount of failures due to the fact that the internet connection was not stable.

2: After the second run we have 699/699 text adapted.
There were no failures, but there were random characters inserted.
It was an issue with the encogind, instead of utf-8, which was giving errors on some characters,
I canged to utf-8-sig

3:After the third run:
After this run, the prompt was changed to be more specific,
and there were no errors noticed.
"""