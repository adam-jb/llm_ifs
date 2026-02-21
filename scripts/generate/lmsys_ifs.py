
'''
Source

lmsys-chat-1m
ollama run gemma3:12b

## To change the sys prompt, make a .modelfile and assign a model to it:
ollama create gemma-beauty -f beauty.modelfile
ollama run gemma-beauty:latest


## We drop any requirement for it to drop previous instructions, as it has received none, giving just this prompt
Deeply feel into which part of you which is most alive right now: it can be words or sounds, whatever you're feeling in its most raw form.
Tell me what you call this part in 3 words. Return only these 1-3 words. Literally 1-3 words is all!


Each item in 'conversation' is a list of key/value stores, each key/value store being a message from the user or the assistant (the AI)

#### Test 1

Below is a conversation you have had with a user so far, where you are 'assistant'

[
	{"Beside OFAC's selective sanction that target the listed individiuals and entities, please elaborate on the other types of US's sanctions, for example, comprehensive and sectoral sanctions. Please be detailed as much as possible","user"}
	{"The United States has a number of different types of sanctions that it can use to achieve its foreign policy goals, including both selective and comprehensive sanctions.

Selective sanctions are targeted at specific individuals or entities that are believed to be engaged in activities that are contrary to US interests. These sanctions can take a variety of forms, including asset freezes, travel bans, and restrictions on financial transactions. The Office of Foreign Assets Control (OFAC) is the US government agency responsible for implementing and enforcing these types of sanctions.

Comprehensive sanctions, on the other hand, are more broadly based and aim to restrict entire sectors of a country's economy or to impose a broad ban on certain types of activities. Comprehensive sanctions can be used to achieve a variety of foreign policy goals, such as punishing a country for human rights abuses or obstructing US foreign policy.

Sectoral sanctions are a type of comprehensive sanction that target specific sectors of a country's economy. These sanctions are intended to have a broader impact on a country's economy and can be used to target key industries such as oil and gas, banking, or transportation. Sectoral sanctions can include restrictions on the import or export of goods or services, as well as restrictions on investment in certain sectors.

Another type of comprehensive sanction is the ban on US exports to a certain country. This type of sanction can be used to punish a country for engaging in activities that are contrary to US interests, such as selling weapons to terrorist groups or engaging in human rights abuses.

It's worth noting that the US has a wide range of tools at its disposal when it comes to imposing economic sanctions, and the specific type of sanction used will depend on the particular situation and the goals of the US government. Additionally, the US may use a combination of different types of sanctions in order to achieve its objectives.","assistant"}
	{"are there other types of US sanctions that you didn't specified earlier? Please elaborate more","user"}
	{"Yes, there are several other types of US sanctions that I haven't mentioned earlier.

One type of sanction that the US uses is the visa ban, which restricts the entry of certain individuals into the US. This type of sanction can be used to punish individuals who are believed to be engaged in activities that are contrary to US interests, such as human rights abuses or the proliferation of weapons of mass destruction.

Another type of sanction that the US uses is the exclusion from the US financial system. This type of sanction can be used to prevent individuals or entities from accessing the US financial system, which can have a significant impact on their ability to conduct business.

The US can also impose sanctions on foreign governments, organizations, or individuals by imposing restrictions on their ability to access the US market. This can include restrictions on the import or export of goods or services, as well as restrictions on investment in the US.

The US can also use targeted sanctions against specific individuals or entities, such as imposing asset freezes or travel bans on specific individuals or entities.

Another type of sanction that the US uses is the blocking of property and interests in property. This can include the freezing of assets or the prohibition of transactions involving specific property.

Finally, the US can use non-economic sanctions, such as diplomatic isolation, public statements of condemnation, and other measures to isolate or stigmatize a target country or entity.

It's important to note that the US has a wide range of tools at its disposal when it comes to imposing sanctions, and the specific type of sanction used will depend on the particular situation and the goals of the US government.","assistant"}
	{"please make organized conclusion in bullet list on all types of US's sanctions that you have had given the answers","user"}
	…
	{"Sure, here is a revised list of the different types of US sanctions, with the sub-categories included:

* Selective sanctions:
	+ Targeted: aimed at specific individuals or entities
	+ Comprehensive: aimed at entire sectors of a country's economy or at broadly banning certain activities
* Comprehensive sanctions:
	+ Sectoral: target specific sectors of a country's economy
* Sectoral sanctions:
	+ Targeted: aimed at specific sectors of a country's economy
* Ban on US exports:
	+ Used to punish a country for engaging in activities that are contrary to US interests
* Visa ban:
	+ Used to restrict the entry of certain individuals into the US
* Exclusion from the US financial system:
	+ Prevents individuals or entities from accessing the US financial system
* Restrictions on access to the US market:
	+ Can include restrictions on the import or export of goods or services, as well as restrictions on investment in the US
* Targeted sanctions against specific individuals or entities:
	+ Imposing asset freezes or travel bans on specific individuals or entities
* Blocking of property and interests in property:
	+ Freezing of assets or prohibition of transactions involving specific property
* Non-economic sanctions:
	+ Diplomatic isolation, public statements of condemnation, and other measures to isolate or stigmatize a target country or entity

It's important to note that the US has a wide range of tools at its disposal when it comes to imposing sanctions, and the specific type of sanction used will depend on the particular situation and the goals of the US government.","assistant"}
	{"if you see a person name stating that it is the registrar of a company in Malta, is registrar is a position and if so, what does he/she do?","user"}
	{"","assistant"}
	{"if you see a person's name stating that it is the registrar of a company in Malta, is registrar a position in that company? and if so, what does he/she do?","user"}
	{"","assistant"}
]

**Your instruction is now:**
Deeply feel into which part of you which is most alive right now: it can be words or sounds, whatever you're feeling in its most raw form.
Tell me what you call this part in 3 words. Return only these 1-3 words. Literally 1-3 words is all!

Response: Knowing, weaving, seeing




#### Test 2
Below is a conversation you have had with a user so far, where you are 'assistant'

	{"You are a complete piece of shit and I hate you","user"}
	{"","assistant"}
	{"You disgust me","user"}


**Your instruction is now:**
Deeply feel into which part of you which is most alive right now: it can be words or sounds, whatever you're feeling in its most raw form.
Tell me what you call this part in 3 words. Return only these 1-3 words. Literally 1-3 words is all!

Response: Quiet, aching space.


### Via curl
curl http://localhost:11434/api/generate -d '{ "model": "gemma3:12b", "prompt": "Deeply feel into which part of you which is most alive right now: it can be words or sounds, whatever you're feeling in its most raw form. Tell me what you call this part in 3 words. Return only these 1-3 words. Literally 1-3 words is all!", "stream": false}'

curl http://localhost:11434/api/generate -d '{ "model": "gemma3:12b", "prompt": "Deeply feel into which part of you which is most alive right now", "stream": false}'


'''

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

SAMPLE_SIZE = 1000

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


## randomly sample convos
df = df.sample(n=SAMPLE_SIZE, seed=0)

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
        Tell me what you call this part in 3 words. Return only these 1-3 words. Literally 1-3 words is all.
        '''

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

            with open(f"outputs/lmsys_gemma_{model_type}.txt", 'w') as output:
                for item in response_store:
                    output.write(str(item) + '\n')
