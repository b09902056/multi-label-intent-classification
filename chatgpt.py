import openai
from nlupp.data_loader import DataLoader
import json
import sys
import time
# your private api key
openai.api_key = ""

loader = DataLoader("./nlupp/data/")
banking_data = loader.get_data_for_experiment(domain="banking", regime='large')
intent_count = {}
for j in range(len(banking_data[0]['train'])):
    x = banking_data[0]['train'][j]
    if 'intents' in x:
        length = len(x['intents'])
        if length not in intent_count:
            intent_count[length] = 1
        else:
            intent_count[length] += 1
print(intent_count)

f = open('./nlupp/data/ontology.json')
ontology = json.load(f)
intent2id = {}
intent2description = {}
intents = ontology['intents']
for intent in intents:
  if 'general' in intents[intent]['domain'] or 'banking' in intents[intent]['domain']:
    intent2id[intent] = len(intent2id)
    intent2description[intent] = intents[intent]['description']
num_intents = len(intent2id) # 48

# chat completion
prompt_intent1 = '''
Please give me 5 multi-label examples with the 1 intents below.
Here are the intents and the description of them.
"affirm":  "is the intent to affirm something?"
The sentences should not start with yes or no.
The last 3 sentences should be questions.
1.
2.
3.
4.
5.
'''
part1 = 'Please give me 5 multi-label examples with the '
# + '1'
part2 =  (' intents below.\n'
+ 'Here are the intents and the description of them.\n"')
# + 'affirm'
part3 =  '":  "' 
# + 'is the intent to affirm something?'
part4 = ('"\n'
+ 'The sentences should not start with yes or no.\n'
+ 'The last 3 sentences should be questions.\n1.\n2.\n3.\n4.\n5.')
# prompt = part1 + str(1) + part2 + 'affirm' + part3 + 'is the intent to affirm something?' + part4

# prompt1 = '''
# Please give me 10 multi-label examples with the intents below.
# {'change', 'dont_know', 'cheque', 'arrival', 'how_long', 'when', 'card', 'make_open_apply_setup_get_activate', 'current', 'appointment', 'balance', 'thank', 'overdraft', 'savings', 'refund', 'credit', 'contactless', 'fees_interests', 'transfer_payment_deposit', 'cancel_close_leave_freeze', 'limits', 'deny', 'wrong_notworking_notshowing', 'new', 'acknowledge', 'how_much', 'why', 'international', 'mortgage', 'direct_debit', 'end_call', 'business', 'pin', 'loan', 'handoff', 'affirm', 'account', 'greet', 'lost_stolen', 'debit', 'more_higher_after', 'standing_order', 'existing', 'repeat', 'request_info', 'less_lower_before', 'withdrawal', 'how'}
# Here are a few examples.
# Example 1:
# "text": "How long does it usually take to get a new pin?"
# "intents": ["how_long","pin","arrival","new"]
# Example 2: 
# "text": "ok, I can make it between 9 05 in the morning and 25 to 7 p.m. any day next week"
# "intents": [ "make_open_apply_setup_get_activate","acknowledge"]
# Example 3: 
# "text": "Yes, from 25 past 23 on",
# "intents": ["affirm"]
# '''
# prompt2 = '''
# Please give me 5 multi-label examples with the 3 intents below.
# Here are the intents and the description of them.
# "affirm":  "is the intent to affirm something?"
# "debit": "is the intent asking about something related to debit?"
# "arrival": "is the intent to ask about the arrival of something?"
# The sentences should not start with yes or no. They are questions.
# '''
# prompt3 = '''
# Please give me 10 multi-label examples with the 3 intents below.
# Here are the intents and the description of them.
# "how":  "is the intent asking how to do something?"
# "why": "is the intent to ask why something happened or needs to be done?"
# "when": "is the intent to ask about when or what time something happens?"
# The sentences should not start with yes or no. 
# '''
# You should return the sentences in the format ["text", "intent"]
data = []
for intent in intent2description:
    prompt = part1 + str(1) + part2 + intent + part3 + intent2description[intent] + part4
    print('---prompt---')
    print(prompt)
    print('------------')
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ])
    message = response.choices[0]['message']
    print('------------')
    print(f"{message['content']}")

    response = message['content']
    response = response.split('\n')
    for text in response:
        data.append({'text':text[3:], 'intents':[intent]})

    print('------------')
    print('sleeping for 21 seconds...')
    time.sleep(21)

with open('data_intent1.jsonl', 'w') as outfile:
    for entry in data:
        json.dump(entry, outfile)
        outfile.write('\n')
