import json
import openai
from tqdm import trange
import random
import time

items = ['banking', 'hotels']

openai.api_key = ''
NUM_DATA_PER_BATCH = 5
TOTAL_BATCH = 20

# Collect descriptions of intents
with open('nlupp/data/ontology.json') as f:
    data = json.load(f)
    with open('nlupp/chatGPT_data/ontology.json', 'w') as f2:
        json.dump(data, f2, indent=4)

    data = data['intents']
    intent_des = {
        'banking': {},
        'hotels': {}
    }
    for i in data.keys():
        if data[i]['domain'][0] == 'general':
            intent_des['banking'][i] = data[i]["description"]
            intent_des['hotels'][i] = data[i]["description"]
        elif data[i]['domain'][0] == 'banking':
            intent_des['banking'][i] = data[i]["description"]
        else:
            intent_des['hotels'][i] = data[i]["description"]

descriptions = {
    'banking': 'Here are the descriptions of the intents:',
    'hotels': 'Here are the descriptions of the intents:'
}
for item in items:
    for i in intent_des[item]:
        descriptions[item] += f'\n\"{i}\": {intent_des[item][i]}'

for item in items:
    for i in trange(20):
        with open(f'nlupp/data/{item}/fold{i}.json') as f:
            data = json.load(f)

        intents, texts = [], []
        for j in data:
            texts.append(j["text"])
            if 'intents' in j.keys():
                intents.append(j["intents"])
            else:
                intents.append([])

        with open(f'nlupp/chatGPT_data/{item}/fold{i}.json') as f:
            temp_data = json.load(f)
        
        for j in trange(TOTAL_BATCH - len(temp_data) // NUM_DATA_PER_BATCH):
            pick = random.sample(range(0, len(data)),  5)

            while 1:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "assistant", "content": descriptions[item]},
                        {"role": "assistant", "content": f"""Here are 5 examples of texts with multiple intents.
                        Example 1:
                        text: {texts[pick[0]]}
                        intents: {intents[pick[0]]}
                        Example 2:
                        text: {texts[pick[1]]}
                        intents: {intents[pick[1]]}
                        Example 3:
                        text: {texts[pick[2]]}
                        intents: {intents[pick[2]]}
                        Example 4:
                        text: {texts[pick[3]]}
                        intents: {intents[pick[3]]}
                        Example 5:
                        text: {texts[pick[4]]}
                        intents: {intents[pick[4]]}
                        """},
                        {"role": "user", "content": f"""Generate five texts with some randomly picked intents. Follow the following format:
                        text:
                        intents:"""},
                    ]
                )
                lines = response['choices'][0]['message']['content'].split('\n')
                temp_text, temp_intent = [], []
                for k in range(len(lines)):
                    s = lines[k].strip()
                    if s.lower().find('text:') != -1:
                        temp_text.append(s[s.lower().find('text:') + 5:].strip(' \"\''))
                        s = lines[k+1].strip()
                        intent = list(s[s.lower().find('intents:') + 8:].strip(' []').split(','))
                        intent = [ii.strip(' \'\"') for ii in intent]
                        temp_intent.append(intent)

                ok = True
                for k in temp_intent:
                    for l in k:
                        if l not in intent_des[item].keys():
                            ok = False

                if ok: break


            #print('-' * 20)
            #print(response['choices'][0]['message']['content'])
            
            for k in range(len(temp_text)):
                temp_data.append(
                    {
                        'text': temp_text[k],
                        'intents': temp_intent[k]
                    }
                )

            random.shuffle(temp_data)
            with open(f'nlupp/chatGPT_data/{item}/fold{i}.json', 'w') as f:
                json.dump(temp_data, f, indent=4)

        print(f'finish {item}: fold {i}')