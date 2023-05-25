import json
import openai
from tqdm import trange
import random

items = ['banking', 'hotels']

openai.api_key = ''

# Collect descriptions of intents
with open('nlupp/data/ontology.json') as f:
    data = json.load(f)
    with open('nlupp/chatGPT_data/ontology.json', 'w') as f2:
        json.dump(data, f2, indent=4)

    data = data['intents']
    intent_des = {}
    for i in data.keys():
        intent_des[i] = data[i]["description"]

for item in items:
    intents, texts = [], []
    for i in range(20):
        with open(f'nlupp/data/{item}/fold{i}.json') as f:
            data = json.load(f)

        for j in data:
            texts.append(j["text"])
            if 'intents' in j.keys():
                intents.append(j["intents"])
            else:
                intents.append([])

    with open(f'{item}_temp.json') as f:
        new_data = json.load(f)

    start_ind = len(new_data) // 5
    for i in trange(start_ind, len(intents)):
        descriptions = 'Here are the descriptions of the intents:'
        for j in intents[i]:
            descriptions += f'\n{j}: {intent_des[j]}'
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "assistant", "content": descriptions},
                {"role": "assistant", "content": f"""Here is an example of sentences with multiple intents:
                "text": "{texts[i]}"
                "intents": {intents[i]}"""},
                {"role": "user", "content": f"""Give me five sentences with multiple intents: {intents[i]}. Do not repeat the intents. Follow the following format:
                1.
                2.
                3.
                4.
                5."""},
            ]
        )
        cnt=0
        for s in response['choices'][0]['message']['content'].split('\n'):
            if s.strip()[:1].isnumeric() and s.strip()[1] == '.':
                new_data.append({
                    "text": s.strip()[2:].strip(' \"\''),
                    "intents": intents[i]
                })
                cnt+=1

        if cnt!=5:
            print(response['choices'][0]['message']['content'])
    
        with open(f'{item}_temp.json', 'w') as f:
            json.dump(new_data, f, indent=4)

    random.shuffle(new_data)

    for i in range(20):
        with open(f'nlupp/chatGPT_data/{item}/fold{i}.json', 'w') as f:
            l, r = len(new_data) // 20 * i, min(len(new_data) // 20 * (i + 1), len(new_data))
            json.dump(new_data[l:r], f, indent=4)

