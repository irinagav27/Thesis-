"""
This code is meant to generate lexically simfilied version of the initial curatorial texts.
The more "complicated" art words and jargon will be changed for more simple words.
While doing so, the LLM has to consider the following: grammar, sentence structure,
and all linguistic features.
What case will this test:
* Impact of vocabulary complexity alone on curatorial voice
* Minimum intervention approach to complexity
* Whether replacing simple words affects the authority of the curatorial voice
* Preservation of epistemic stance and scholarly markers
"""

import pandas as pd
from openai import OpenAI
from httpx import Timeout
from dotenv import load_dotenv
import os
import time
from tqdm import tqdm



load_dotenv('Config/api_keys.env') #Loading the API key

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
    "this function is a helper function, which has the purpose of calling the"
    "Deepseek API, and retry automatically if it fails. It can fail up to 8 times"
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
            wait_time = min(5 * (2 ** attempt), 60) # calculates waiting time before trying again
            #The wait_time is called an exponential backoff, meaning each waiting time increases exponentially 
            #With each retry up to 60 seconds

            print(f"\nRetry {attempt + 1}/{max_retries}")
            print(f"Error: {e}")
            print(f"Waiting {wait_time} seconds...\n")

            time.sleep(wait_time)

    raise Exception("Max retries exceeded")


print("Loading lexical simplification prompt...")

with open('Prompts/lexical_prompt.txt', 'r', encoding='utf-8') as f: #Loading the prompt for the Lexical Adaptation
    PROMPT_TEMPLATE = f.read()
    #In this case a prompt file is used, as if it is needed to change and adapt the prompt
    #It is easier to change the prompt file, rather than changing the code itself

#Loading the dataset of curatorial texts
print("Loading dataset...")

df = pd.read_excel('Data/Original/Dataset_Curatorial_Text.xlsx') #This is the input
#Based on the collection of curatorial texts
#The data that will be taken from this are the unique IDs for each individual text, and the cruatorial text
#These two column are: ID and CuratorialText

print(f"Found {len(df)} texts to process\n")


#Where the output of the generated simplified will be
#In case there is no such folder for the output, one will be created
os.makedirs('Data/adapted/Lexical_Simplification', exist_ok=True)
"""There will be three files for output:
    the checkpoint file: This file is meant to monitor the intermediate results, and to help track how to code is running.
    The final output file: This file is meant to store all the final results of the succesfully adpated texts
    The failed_output file: This file is also a checkpoint file, but differently from the first one, it only saves the IDs of the curatorial texts that could not be adapted
"""


checkpoint_path = 'Data/adapted/Lexical_Simplification/lexical_simplified_checkpoint.csv'
final_output_path = 'Data/adapted/Lexical_Simplification/lexical_simplified.csv'
failed_output_path = 'Data/adapted/Lexical_Simplification/failed_rows.csv'

# In case a checkpoint exist, to note it. It is meant to appear every 5 texts.
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
For each curatorial text, this loop will create an output of the lexically simplified version
of the initial text, and it will also keep track of the word counts of the original and
of the adapted texts.

"""
for idx, row in tqdm(df.iterrows(),
                     total=len(df),
                     desc="Lexical Simplification"):

    text_id = row['ID']

    # Skip already processed rows
    if text_id in processed_ids:
        continue

    original_text = row['CuratorialText']
    # This is meant to deal with empty texts, although there are not empty texts
    #in the dataset, I wanted this case covered too.
    if pd.isna(original_text) or str(original_text).strip() == "":

        print(f"Skipping {text_id} - empty text")

        results.append({
            'ID': text_id,
            'CuratorialText': original_text,
            'lexical_adapted': None,
            'word_count_original': 0,
            'word_count_adapted': 0,
            'error': 'Empty original text'
        })

        continue

#Crating the prompt.
    full_prompt = f"""
{PROMPT_TEMPLATE}

Original text:
\"\"\"
{original_text}
\"\"\"

Provide the lexically simplified version:
"""

    try:

        messages = [
            {
                "role": "system",
                "content": "You are a lexical simplification expert for museum curatorial texts."
            },
            {
                "role": "user",
                "content": full_prompt
            }
        ]

        response = call_deepseek_with_retry(messages)

        adapted_text = response.choices[0].message.content.strip()

        # Storing the results
        results.append({
            'ID': text_id,
            'CuratorialText': original_text,
            'lexical_adapted': adapted_text,
            'word_count_original': len(str(original_text).split()),
            'word_count_adapted': len(adapted_text.split()),
            'error': None
        })


        if len(results) % 5 == 0:

            pd.DataFrame(results).to_csv(
                checkpoint_path,
                index=False,
                encoding='utf-8-sig' # This is the encoding that was used in this case
                #As there was an issue with corrupted characters in the output
                #If the encoding was set to utf-8
                
            )

            print(f" Checkpoint saved ({len(results)} rows)")


        time.sleep(1) # the rate limiting


# In case there is an error
#This is made as such if there is one error row, not the entire code will fail
#And only one row will be lost.
    except Exception as e:

        print(f"\n Error processing {text_id}: {e}")

        results.append({
            'ID': text_id,
            'CuratorialText': original_text,
            'lexical_adapted': None,
            'word_count_original': len(str(original_text).split()),
            'word_count_adapted': 0,
            'error': str(e)
        })

        # Save checkpoint even after failures
        pd.DataFrame(results).to_csv(
            checkpoint_path,
            index=False,
            encoding='utf-8-sig'
            
        )

        time.sleep(3)


#Saving the final results
print("\nSaving final results...")

df_results = pd.DataFrame(results)

df_results.to_csv(
    final_output_path,
    index=False,
    encoding='utf-8-sig'
)

# In case there are failed rows, it is meant to keep track of all the texts lost
failed_rows = df_results[df_results['error'].notna()]

if len(failed_rows) > 0:

    failed_rows.to_csv(
        failed_output_path,
        index=False,
        encoding='utf-8-sig'
    )

    print("\nFailed IDs:")
    print(failed_rows['ID'].tolist())


# Summary of the run
successful = df_results['error'].isna().sum()
failed = len(df_results) - successful

print("\n" + "=" * 60)
print("LEXICAL SIMPLIFICATION COMPLETE")
print("=" * 60)
print(f" Successful: {successful}/{len(df_results)}")
print(f" Failed: {failed}/{len(df_results)}")
print(f" Final results: {final_output_path}")
print(f" Checkpoint file: {checkpoint_path}")

if failed > 0:
    print(f" Failed rows: {failed_output_path}")

print("=" * 60)

""""
Run 1,2 and 3: The first two runs copy-pasted the text instead of adapting it.
The third run started to work better.
4:
* 12 missing texts. The missing texts: 
['CT0126', 'CT0127', 'CT0128', 'CT0129', 'CT0130',
'CT0131', 'CT0132', 'CT0133', 'CT0134', 'CT0244', 'CT0245', 'CT0246'](can be found within the file failed_rows.csv)
5: 
*After the 5th run there were no more missing text or ignored texts
*But there were issues with random character being inserted instead of kommas and apostrophes
*example for as such: This exhibition marks the first time Matisseâ€™s Jazz will be shown in its 
entirety at the Art Institute; the museum acquired it in 1948. 
One of the most important artist books of the 20th century, 
Jazz joins over 50 works from the museumâ€™s famous collection of paintings, 
sculpture, drawings, prints, illustrated books, and textiles to showcase the 
well-known artistâ€™s commitment across his 50-year career to constant new ideas 
and the expressive power of color and line.
* as such the encoding was chagned to utf-8-sig, instead of utf-8
* this managed to fix the issue of random characters
after chaning the prompts:
1.1:
* the prompt was changed to be more specific, as I noticed that in the initial prompts,
i forgot to includ specific keywords such as "rules" or "output" 
which might have led to the model not fully understanding the task
* After the change, the model started to produce slightly better results
1.2: More simplified prompts:

"""