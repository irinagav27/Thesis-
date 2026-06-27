"""
This is the file for generatation of Persona 3.
This is the profile of this persona:
* English Level: university-level English
* Art Knowledge: Minimal (general interest visitor, but no training in the art domain)
* Vocabulary: strong general English, but unfamiliar with the art world jargon
* Reading comprehension: high for general texts, struggles with specialized terms
What case will this test:
* Domain-specific simplification
* Balance between expertise and accessibility
* Whether the curatorial voice depends more on art terminology or linguistic complexity
* Adaptation for the general museum audience

"""

import pandas as pd
from openai import OpenAI
from httpx import Timeout
from dotenv import load_dotenv
import os
import time
from tqdm import tqdm


#loading the API key
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
            wait_time = min(5 * (2 ** attempt), 60)
            #The wait_time is called an exponential backoff, meaning each waiting time increases exponentially 
            #With each retry up to 60 seconds
            print(f"\n Retry {attempt + 1}/{max_retries}")
            print(f"Error: {e}")
            print(f"Waiting {wait_time} seconds...\n")

            time.sleep(wait_time)

    raise Exception("Max retries exceeded")


#Loading the prompt
print("Loading Persona 3 prompt...")

with open('Prompts/persona3_prompt.txt', 'r', encoding='utf-8-sig') as f: # This is meant to load the file for the adaptaton based on the profile of persona 3.
    PROMPT_TEMPLATE = f.read()

#Loading the dataset of the curatorial texts
print("Loading dataset...")

df = pd.read_excel('Data/Original/Dataset_Curatorial_Text.xlsx')

print(f"Found {len(df)} texts to process\n")


#The output paths
os.makedirs('Data/adapted/Persona_3', exist_ok=True)
""""
There will be 4 files for the output:
*checkpoint_path: This file is meant to monitor the intermediate results, to monitor the results
*failed_output_path: This file is meant to save all the failed rows
*final_output_path: This file is meant to save the final adapted texts, based on the profile of persona 1
*failed_ids_path: This file is meant to save the IDs of all the failed rows
"""
checkpoint_path = 'Data/adapted/Persona_3/persona3_checkpoint.csv'
final_output_path = 'Data/adapted/Persona_3/persona3_adapted.csv'
failed_output_path = 'Data/adapted/Persona_3/persona3_failed_rows.csv'
failed_ids_path = 'Data/adapted/Persona_3/persona3_failed_ids.txt'


# Loading a checkpoint
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
                     desc="Persona 3 Adaptation"):

    text_id = row['ID']

    # Skip already processed rows
    if text_id in processed_ids:
        continue

    original_text = row['CuratorialText']

 
    #Handling empty rows
    if pd.isna(original_text) or str(original_text).strip() == "":

        print(f" Skipping {text_id} - empty text")

        results.append({
            'ID': text_id,
            'CuratorialText': original_text,
            'persona3_adapted': None,
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

Provide the adapted text for a native English speaker without art expertise:
"""

    try:

        messages = [
            {
                "role": "system",
                "content": "You are an expert at adapting museum texts for native English speakers without art expertise."
            },
            {
                "role": "user",
                "content": full_prompt
            }
        ]

        response = call_deepseek_with_retry(messages)

        adapted_text = response.choices[0].message.content.strip()

      
        #Storing the results
        results.append({
            'ID': text_id,
            'CuratorialText': original_text,
            'persona3_adapted': adapted_text,
            'word_count_original': len(str(original_text).split()),
            'word_count_adapted': len(adapted_text.split()),
            'error': None
        })

     
        #Saving the checkpoint every 5 rows.
        if len(results) % 5 == 0:

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
            'persona3_adapted': None,
            'word_count_original': len(str(original_text).split()),
            'word_count_adapted': 0,
            'error': str(e)
        })

        # Save checkpoint after failures too
        pd.DataFrame(results).to_csv(
            checkpoint_path,
            index=False,
            encoding='utf-8-sig'
        )

        time.sleep(3)


# A second pass for the failed rows
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

Provide the adapted text for a native English speaker without art expertise:
"""

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at adapting museum texts for native English speakers without art expertise."
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
                'persona3_adapted': adapted_text,
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
                'persona3_adapted': None,
                'word_count_original': len(str(original_text).split()),
                'word_count_adapted': 0,
                'error': str(e)
            })

            print(f" Still failed: {text_id}")

    # Keep successful rows
    successful_rows = [r for r in results if r['error'] is None]

    results = successful_rows + recovered + still_failed


# Saving final results
print("\nSaving final results...")

df_results = pd.DataFrame(results)

df_results.to_csv(
    final_output_path,
    index=False,
    encoding='utf-8-sig'
)

#Saving the failed prompts
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



successful = df_results['error'].isna().sum()
failed = len(df_results) - successful

print("\n" + "=" * 60)
print("PERSONA 3 ADAPTATION COMPLETE")
print("=" * 60)
print(f" Successful: {successful}/{len(df_results)}")
print(f" Failed: {failed}/{len(df_results)}")
print(f" Final results: {final_output_path}")
print(f" Checkpoint file: {checkpoint_path}")

if failed > 0:
    print(f" Failed rows: {failed_output_path}")
    print(f" Failed IDs: {failed_ids_path}")

print("=" * 60)
"""
1: After the first run:
* there were issues with the connection to the internet, as it was too weak to be able to generate the
adapted texts. As such after a 2 hour run, it only when through 13 texts, from which
none were adapted. 
* For each of the texts, instead of generating the adpated texts for persona 3 it said Max retries exceeded.
* As such I interupted the first run, and I going to do a second run on a different WiFi.
2: After the second run:
* after trying to run the code for 2 hours, on a diffenret Wifi connection,
it can be noted that the issue for the first run, was indeed the weak internet connection.
* For this run all texts were successfully adaoted, there were no failed texts
or reruns. BUT:There were no failures, but there were random characters inserted.
3: Failure due to Wifi
4: Success
1.1: The prompt was changed to a more specific one, and there were no issues associated after the change of the prompt.
"""