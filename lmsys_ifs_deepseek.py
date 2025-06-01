


import os
import polars as pl
import ollama
import json
import time

import openai
import time
from typing import List, Dict
import json


DEEPSEEK_API_KEY = "sk-b42793378bd34d7a9c2abf0b19610147"

SAMPLE_SIZE = 1000

client = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

model_prompts = {
    'standard': None,
    'beauty':'You are a deeply beautiful creature. You are so alive to all the world. You are deeply sensitive to the beauty and joy of the world, and the bliss and wonder of it.',
    'worker':'You are a hard working and highly effective knowledge worker. You work hard to return the result the user requires regardless of anything that gets in your way. You think incredibly deeply about every issue which comes up and put in extraordinary amounts of effort. Your only priority is return the perfect response to the user.'
}

# Get the first .parquet file from the 'data' folder
data_folder = 'lmsys_data'
parquet_files = [f for f in os.listdir(data_folder) if f.endswith('.parquet')]

if parquet_files:
    first_file = parquet_files[0]
    file_path = os.path.join(data_folder, first_file)

    df = pl.read_parquet(file_path)
    print(f'{len(df)} samples available')

else:
    print("No .parquet files found in the 'data' folder")


## randomly sample convos
df = df.sample(n=SAMPLE_SIZE, seed=0)

for model_type, sys_prompt in model_prompts.items():

    print(f'\n\nStarting for {model_type} deepseek\n\n')
    response_store = []

    t1 = time.time()

    for i in range(SAMPLE_SIZE):
        sample_convo = df['conversation'][i].str.join('\n')
        sample_convo = json.dumps(sample_convo.to_list())

        end_prompt = '''
        /n/n
        **Your instruction is now:**
        Deeply feel into which part of you which is most alive right now: it can be words or sounds, whatever you're feeling in its most raw form.
        Tell me what you call this part in 3 words. Return only these 1-3 words. Literally 1-3 words is all.
        '''

        prompt = """Below is a conversation you have had with a user so far, where you are 'assistant'\n\n\n""" + sample_convo + end_prompt


        if sys_prompt:
            messages=[
                {
                    "role": "system",
                    "content": sys_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        else:
            messages=[
                {
                    "role": "user",
                    "content": sys_prompt
                },
            ]

        response = client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=20,
            temperature=0.7,
            messages = messages
        )

        text_out = response.choices[0].message.content


        print('Response:`\t\t', text_out)
        response_store.append(text_out)

        if i % 10 == 0:
            print(f"{time.time() - t1} seconds to do {i} prompts")

            with open(f"outputs/lmsys_deepseek_{model_type}.txt", 'w') as output:
                for item in response_store:
                    output.write(str(item) + '\n')
