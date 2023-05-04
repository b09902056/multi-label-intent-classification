import pandas as pd
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
import random
import json
from nlupp.data_loader import DataLoader
import sys
from sklearn.metrics import classification_report, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')
import argparse

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--b", help = "batch size", type=int, default=8)
parser.add_argument("--e", help = "epoch", type=int, default=50)
parser.add_argument("--lr", help = "learning rate", type=float, default=1e-4)
parser.add_argument("--regime", help = "regime", type=str, default='mid')
parser.add_argument("--fold", help = "fold", type=int, default=None)
parser.add_argument("--domain", help = "domain", type=str, default="banking")
args = parser.parse_args()

hidden_size = 768 # roberta-base = 768, roberta-large=1024
dropout = 0.5
batch_size = args.b
EPOCHS = args.e
LR = args.lr
threshold = 0.5
regime = args.regime
pretrained_model = 'roberta-base'
domain = args.domain
print(f'dropout={dropout}, batch_size={batch_size}, epoch={EPOCHS}, LR={LR}, regime={regime}')

f = open('./nlupp/data/ontology.json')
ontology = json.load(f)
label2id = {}
intents = ontology['intents']
for intent in intents:
    if domain == "banking":
        if 'general' in intents[intent]['domain'] or 'banking' in intents[intent]['domain']:
            label2id[intent] = len(label2id)
    elif domain == "hotels":
        if 'general' in intents[intent]['domain'] or 'hotels' in intents[intent]['domain']:
            label2id[intent] = len(label2id)
    else:
        label2id[intent] = len(label2id)
        
num_intents = len(label2id) # 48

#loader = DataLoader("./nlupp/data/")
ori_loader = DataLoader("./nlupp/data/")
aug_loader = DataLoader("./nlupp/chatGPT_data/")

ori_data = ori_loader.get_data_for_experiment(domain, regime=regime)
aug_data = aug_loader.get_data_for_experiment(domain, regime=regime)
fold = len(ori_data)
if args.fold != None:
    fold = args.fold
print('folds =', fold)

def same_seed(seed):
    random.seed(seed)
    np.random.seed(seed)  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seed(56)


tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)

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

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)


        return linear_output

def train(model, train_data, val_data, learning_rate, epochs):

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('device =', device)

    criterion = nn.BCELoss()
    optimizer = AdamW(model.parameters(), lr= learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    best_acc = 0
    for epoch_num in range(epochs):
        total_loss_train = 0
        train_labels = []
        train_output = []

        for train_input, train_label in tqdm(train_dataloader):
            train_labels.extend(train_label)
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            output = torch.sigmoid(output)
            
            batch_loss = criterion(output, train_label.float())
            total_loss_train += batch_loss.item()
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            for i in range(len(output)):
                for j in range(len(output[i])):
                    if (output[i][j] > threshold):
                        output[i][j] = 1
                    else:
                        output[i][j] = 0
            train_output.extend(output)

        
        total_loss_val = 0
        eval_labels = []
        eval_output = []
        with torch.no_grad():
            for val_input, val_label in tqdm(val_dataloader):
                eval_labels.extend(val_label)
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                output = torch.sigmoid(output)

                batch_loss = criterion(output, val_label.float())
                total_loss_val += batch_loss.item()
                
                for i in range(len(output)):
                    for j in range(len(output[i])):
                        if (output[i][j] > threshold):
                            output[i][j] = 1
                        else:
                            output[i][j] = 0
                eval_output.extend(output)

        train_labels = [label.cpu().detach().numpy() for label in train_labels]
        train_output = [output.cpu().detach().numpy() for output in train_output]
        eval_labels = [label.cpu().detach().numpy() for label in eval_labels]
        eval_output = [output.cpu().detach().numpy() for output in eval_output]
        print(f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data["text"]): .3f} | Val Loss: {total_loss_val / len(test_data["text"]): .3f}')
        train_accuracy = accuracy_score(train_labels, train_output)
        train_f1_score_micro = f1_score(train_labels, train_output, average='micro')
        train_f1_score_macro = f1_score(train_labels, train_output, average='macro')
        print(f"train Accuracy Score = {train_accuracy}", end = ', ')
        print(f"train F1 Score (Micro) = {train_f1_score_micro}", end = ', ')
        print(f"train F1 Score (Macro) = {train_f1_score_macro}")
        eval_accuracy = accuracy_score(eval_labels, eval_output)
        eval_f1_score_micro = f1_score(eval_labels, eval_output, average='micro')
        eval_f1_score_macro = f1_score(eval_labels, eval_output, average='macro')
        print(f"eval Accuracy Score = {eval_accuracy}", end = ', ')
        print(f"eval F1 Score (Micro) = {eval_f1_score_micro}", end = ', ')
        print(f"eval F1 Score (Macro) = {eval_f1_score_macro}")
        #print(classification_report(eval_labels, eval_output))

        #if eval_accuracy >= best_acc:
            #best_acc = eval_accuracy
            #torch.save(model.state_dict(), "./intent_model.ckpt")
            #print(f'save model: eval acc = {eval_accuracy}')

def evaluate(model, test_data):

    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #ckpt = torch.load('./intent_model.ckpt')
    #model.load_state_dict(ckpt)

    if use_cuda:
        model = model.cuda()

    with torch.no_grad():
        test_labels = []
        test_output = []
        for test_input, test_label in tqdm(test_dataloader):
            test_labels.extend(test_label)
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            output = torch.sigmoid(output)

            for i in range(len(output)):
                for j in range(len(output[i])):
                    if (output[i][j] > threshold):
                        output[i][j] = 1
                    else:
                        output[i][j] = 0
            test_output.extend(output)
        test_labels = [label.cpu().detach().numpy() for label in test_labels]
        test_output = [output.cpu().detach().numpy() for output in test_output]
        test_accuracy = accuracy_score(test_labels, test_output)
        test_f1_score_micro = f1_score(test_labels, test_output, average='micro')
        test_f1_score_macro = f1_score(test_labels, test_output, average='macro')
        print(f"test Accuracy Score = {test_accuracy}", end = ', ')
        print(f"test F1 Score (Micro) = {test_f1_score_micro}", end = ', ')
        print(f"test F1 Score (Macro) = {test_f1_score_macro}")

        #report = classification_report(test_labels, test_output, output_dict=True)
        return test_accuracy, test_f1_score_micro
                  

total_accuracy = 0
total_f1 = 0
#total_report = pd.DataFrame()

for i in range(fold):
    print('fold', i)
    train_data = {'text':[], 'intents':[], 'labels':[]}
    eval_data = {'text':[], 'intents':[], 'labels':[]}
    test_data = {'text':[], 'intents':[], 'labels':[]}

    # add original data to test data
    for j in range(len(ori_data[i]['test'])):
        x = ori_data[i]['test'][j]
        if 'intents' not in x: x['intents'] = []
        test_data['text'].append(x['text'])
        test_data['intents'].append(x['intents'])
        labels = [0] * num_intents
        for intent in x['intents']:
            labels[label2id[intent]] = 1
        test_data['labels'].append(labels)

    # add augmented data to train data
    for j in range(len(aug_data[i]['train'])):
        x = aug_data[i]['train'][j]
        if 'intents' not in x: x['intents'] = []
        train_data['text'].append(x['text'])
        train_data['intents'].append(x['intents'])
        labels = [0] * num_intents
        for intent in x['intents']:
            labels[label2id[intent]] = 1
        train_data['labels'].append(labels)

    # add original data to train data
    for j in range(len(ori_data[i]['train'])):
        x = ori_data[i]['train'][j]
        if 'intents' not in x: x['intents'] = []
        train_data['text'].append(x['text'])
        train_data['intents'].append(x['intents'])
        labels = [0] * num_intents
        for intent in x['intents']:
            labels[label2id[intent]] = 1
        train_data['labels'].append(labels)

    print('train len =', len(train_data['text'])) 
    print('test len =', len(test_data['text'])) 

    model = BertClassifier(hidden_size, dropout)
    train(model, train_data, test_data, LR, EPOCHS)

    test_accuracy, test_f1 = evaluate(model, test_data)
    total_accuracy += test_accuracy
    total_f1 += test_f1

print('---------------------')
print(f'dropout={dropout}, batch_size={batch_size}, epoch={EPOCHS}, LR={LR}, regime={regime}, fold={fold}')
print(f'total accuracy = {total_accuracy / fold}')
print(f'total f1 = {total_f1 / fold}')
print('---------------------')