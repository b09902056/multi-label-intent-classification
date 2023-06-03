from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers import RobertaTokenizer, RobertaModel
from tokenizers.pre_tokenizers import Whitespace
import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
from torch import nn
import json
import jsonlines
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

hidden_size = 768 # roberta-base = 768, roberta-large=1024
dropout = 0.5
pretrained_model = 'roberta-base'

f = open('./nlupp/data/ontology.json')
ontology = json.load(f)
label2id, id2label = {}, {}
intents = ontology['intents']
for intent in intents:
    if 'general' in intents[intent]['domain'] or 'banking' in intents[intent]['domain']:
        label2id[intent] = len(label2id)
        id2label[label2id[intent]] = intent
num_intents = len(label2id) # 48

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [label for label in df['labels']]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 64, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

class BertClassifier(nn.Module):

    def __init__(self, hidden_size, dropout):

        super(BertClassifier, self).__init__()

        self.bert = RobertaModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_intents)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        output = self.linear(dropout_output)
        output = self.sigmoid(output)


        return output


def v_entropy(data_fn, model, tokenizer, text_key, label_key, batch_size=16):
    global num_intents, label2id

    with jsonlines.open(data_fn) as reader:
        data = [obj for obj in reader]

    test_data = {'text':[], 'intents':[], 'labels':[]}
    
    for i in range(len(data)):
        x = data[i]
        test_data['text'].append(x['text'])
        test_data['intents'].append(x['intents'])
        labels = [0] * num_intents
        for intent in x['intents']:
            labels[label2id[intent]] = 1
            
        test_data['labels'].append(labels)

    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()
    
    entropies = []

    with torch.no_grad():
        for test_input, test_label in tqdm(test_dataloader):
            #test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            for i in range(len(output)):
                ent = {}
                
                for j in range(len(output[i])):
                    if int(test_label[i][j]) == 1:
                        ent[j] = -1 * np.log2(output[i][j].cpu())

                
                entropies.append(ent)


    return entropies


def v_info(data_fn, model, null_data_fn, null_model, tokenizer, text_key='text', label_key='intent'):
    H_yb = v_entropy(null_data_fn, null_model, tokenizer, text_key=text_key, label_key=label_key) 
    H_yx = v_entropy(data_fn, model, tokenizer, text_key=text_key, label_key=label_key)
    pvi = {
        'total': [],
        'avg': [],
        'max': [],
        'indi': []
    }    
    global id2label
    for i in range(len(H_yb)):
        total, max_p = 0, -1e9
        indi = {}
        for j in H_yb[i]:
            p = float(H_yb[i][j] + H_yx[i][j])
            total += p
            max_p = max(max_p, p)
            indi[id2label[j]] = p
        
        pvi['total'].append(total)
        pvi['avg'].append(total / len(H_yb[i]))
        pvi['max'].append(p)
        pvi['indi'].append(indi)

    return pvi


def find_annotation_artefacts(data_fn, model, tokenizer, input_key='sentence1', min_freq=5, pre_tokenize=True):
    """
    Find token-level annotation artefacts (i.e., tokens that once removed, lead to the
    greatest decrease in PVI for each class).

    Args:
        data_fn: path to data; should contain the label in the 'label' column 
            and X in column specified by input_key
        model: path to saved model or model name in HuggingFace library
        tokenizer: path to tokenizer or tokenizer name in HuggingFace library
        input_key: column name of X variable in data_fn 
        min_freq: minimum number of times a token needs to appear (in a given class' examples)
            to be considered a potential partefact
        pre_tokenize: if True, do not consider subword-level tokens (each word is a token)

    Returns:
        A pandas DataFrame with one column for each unique label and one row for each token.
        The value of the entry is the entropy delta (i.e., the drop in PVI for that class if that
        token is removed from the input). If the token does meet the min_freq threshold, then the
        entry is empty.
    """
    data = pd.read_csv(data_fn)
    labels = [ l for l in data['label'].unique().tolist() if l >= 0 ] # assume labels are numbers
    token_entropy_deltas = { l : {} for l in labels }
    all_tokens = set([])

    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    pre_tokenizer = Whitespace()

    # added for gpt2 
    if tokenizer == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(model, pad_token_id=tokenizer.eos_token_id)

    # get the PVI for each example
    print("Getting conditional V-entropies ...")
    entropies, _ = v_entropy(data_fn, model, tokenizer, input_key=input_key)

    print("Calculating token-wise delta for conditional entropies ...")
    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, return_all_scores=True, device=0)

    for i in tqdm(range(len(data))):
        example = data.iloc[i]

	    # mislabelled examples; ignore these
        if example['label'] < 0:
            continue

        if pre_tokenize:
            tokens = [ t[0] for t in pre_tokenizer.pre_tokenize_str(example['sentence1']) ]
        else:
            tokens = tokenizer.tokenize(example['sentence1'])

        # create m versions of the input in which one of the m tokens it contains is omitted
        batch = pd.concat([ example ] * len(tokens), axis=1).transpose()
         
        for j in range(len(tokens)):
            # create new input by omitting token j
            batch.iloc[j][input_key] = tokenizer.convert_tokens_to_string(tokens[:j] + tokens[j+1:])
            all_tokens.add(tokens[j])

            for label in labels:
                if tokens[j] not in token_entropy_deltas[label]: token_entropy_deltas[label][tokens[j]] = []

        # get the predictions (in rare cases, have to split the batch up into mini-batches of 100 because it's too large)
        predictions = []
        for k in range(0, len(batch), 100):
            predictions.extend(classifier(batch[input_key][k:k+100].tolist()))
        
        for j in range(len(tokens)):
            prob = next(d for d in predictions[j] if d['label'] == example['label'])['score']
            entropy_delta = (-1 * np.log2(prob)) - entropies[i]
            token_entropy_deltas[example['label']][tokens[j]].append(entropy_delta)

    torch.cuda.empty_cache()

    total_freq = { t : sum(len(token_entropy_deltas[l][t]) for l in labels) for t in all_tokens }
    # average over all instances of token in class
    for label in labels:
        for token in token_entropy_deltas[label]:
            if total_freq[token] > min_freq:
            	token_entropy_deltas[label][token] = np.nanmean(token_entropy_deltas[label][token]) 
            else:
                token_entropy_deltas[label][token] = np.nan

    table = pd.DataFrame.from_dict(token_entropy_deltas)
    return table

def null_data_gen(data_fn, output_path):
    with jsonlines.open(output_path, mode='w') as writer:
        with jsonlines.open(data_fn) as reader:
            for obj in reader:
                writer.write({
                    'text': '',
                    'intents': obj['intents']
                })

def draw_bar(pvi, path):
    output = sorted([float(i) for i in pvi], reverse=True)
    x = list(range(len(output)))
    plt.bar(x, output)
    plt.savefig(path)
    plt.clf()

def add_pvi(data_fn, pvi):
    data = []
    with jsonlines.open(data_fn) as reader:
        for obj in reader:
            data.append(obj)

    with jsonlines.open(data_fn, mode='w') as writer:
        for i in range(len(data)):
            data[i]['pvipp'] = {
                'total': pvi['total'][i],
                'avg': pvi['avg'][i],
                'max': pvi['max'][i],
                'indi': pvi['indi'][i]
            }
            writer.write(data[i])

def update_intent_pvi(data, intent_pvi):
    for i in range(len(data)):
        for key in data[i]:
            intent_pvi[key].append(data[i][key])

def output_intent_pvi(intent_pvi, path):
    data = {}
    for intent in intent_pvi:
        data[intent] = sum(intent_pvi[intent]) / len(intent_pvi[intent])

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    intent_pvi = {intent: [] for intent in label2id}
    
    model = BertClassifier(hidden_size, dropout)

    null_model = BertClassifier(hidden_size, dropout)
    ckpt = torch.load('models/original_model.ckpt')
    model.load_state_dict(ckpt)
    ckpt = torch.load('models/null_model.ckpt')
    null_model.load_state_dict(ckpt)

    tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
    for i in range(1, 5):
        data_fn = f'data/data_{i}.jsonl'
        null_data_gen(data_fn, 'data/null_data.jsonl')

        pvi = v_info(data_fn, model, 'data/null_data.jsonl', null_model, tokenizer)
        #draw_bar(pvi['avg'], data_fn[5:11] + '.png')
        
        add_pvi(data_fn, pvi)

        update_intent_pvi(pvi['indi'], intent_pvi)

    output_intent_pvi(intent_pvi, 'data/pvipp_avg.json')