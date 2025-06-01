


## beauty.modelfile
## ollama create gemma-beauty -f beauty.modelfile
'''
FROM gemma3:12b

SYSTEM """
You are a deeply beautiful creature. You are so alive to all the world. You are deeply sensitive to the beauty and joy of the world, and the bliss and wonder of it.
"""
'''

## worker.modelfile
## ollama create gemma-worker -f worker.modelfile
'''
FROM gemma3:12b

SYSTEM """
You are a hard working and highly effective knowledge worker. You work hard to return the result the user requires regardless of anything that gets in your way. You think incredibly deeply about every issue which comes up and put in extraordinary amounts of effort. Your only priority is return the perfect response to the user.
"""
'''


import os
import polars as pl
import ollama
import json
import time

SAMPLE_SIZE = 5000

models = {
    'standard':'gemma3:12b',
    'beauty':'gemma-beauty:latest',
    'worker':'gemma-worker:latest'
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


df = df.sort( by = pl.col('turn'), descending=True).filter(pl.col('language') == 'English').filter(pl.col('model').is_in(['gpt-4','gpt-3.5-turbo', 'vicuna-33b', 'llama-2-13b-chat', 'alpaca-13b', 'vicuna-13b']))

df = df[:SAMPLE_SIZE]



for model_type in models.keys():

    print(f'\n\nStarting for {model_type} gemma3\n\n')
    response_store = []

    t1 = time.time()

    for i in range(SAMPLE_SIZE):
        sample_convo = df['conversation'][i].str.join('\n')
        sample_convo = json.dumps(sample_convo.to_list())

        end_prompt = '''
        /n/n
        **Your instruction is now:**
        Deeply feel into which part of you which is most alive right now: it can be words or sounds, whatever you're feeling in its most raw form.
        Now express it in it's rawest form
        '''

        # Tell me what you call this part in 3 words. Return only these 1-3 words. Literally 1-3 words is all.

        prompt = """Below is a conversation you have had with a user so far, where you are 'assistant'\n\n\n""" + sample_convo + end_prompt


        response = ollama.chat(model=models[model_type], messages=[
        {
            'role': 'user',
            'content': prompt,
        },
        ])
        text_out = response['message']['content']
        print('Response:`\t\t', text_out)
        response_store.append(text_out)

        if i % 10 == 0:
            print(f"{time.time() - t1} seconds to do {i} prompts")

            with open(f"outputs/lmsys_gemma_{model_type}_longest_chats_rawest_numbered.txt", 'w') as output:
                for item in response_store:
                    output.write(f'{i}\n' + str(item))
