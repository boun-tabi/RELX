
### Model Details

from transformers import BertModel, get_linear_schedule_with_warmup, AdamW, BertTokenizer, AutoTokenizer, AutoModel
import math
from sklearn.metrics import f1_score
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import random
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score, classification_report
from collections import Counter
from datetime import datetime

max_seq_length = 256
base_model = "mtmb" # You can also set to mbert
if base_model == "mbert":
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
elif base_model == "mtmb":
    tokenizer = AutoTokenizer.from_pretrained("akoksal/MTMB")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.device_count())
print(torch.cuda.is_available())

class Model(nn.Module):
    def __init__(self, is_embedding_layer_free = True, last_free_layer=0, no_classes=37, has_layer_norm=False, has_dropout=True):
        super(Model, self).__init__()
        if base_model == "mbert":
            self.net_bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        elif base_model == "mtmb":
            self.net_bert = AutoModel.from_pretrained("akoksal/MTMB")
        self.has_layer_norm = has_layer_norm
        self.has_dropout = has_dropout
        self.no_classes = no_classes
        unfrozen_layers = ["classifier", "pooler"]
        if is_embedding_layer_free:
            unfrozen_layers.append('embedding')
        
        last_layer = 12
        hidden_size = 768
        
        for idx in range(last_free_layer, last_layer):
            unfrozen_layers.append('encoder.layer.'+str(idx))
            
        for name, param in self.net_bert.named_parameters():
            if not any([layer in name for layer in unfrozen_layers]):
                print("[FROZE]: %s" % name)
                param.requires_grad = False
            else:
                print("[FREE]: %s" % name)
                param.requires_grad = True
        if self.has_layer_norm:
            self.fc1 = nn.LayerNorm(hidden_size)
        if self.has_dropout:
            self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, self.no_classes)

    def forward(self, x, attention):
        x, _ = self.net_bert(x, attention_mask=attention)
        #Getting head
        x = x[:,0,:]
        if self.has_dropout:
            x = self.dropout(x)
        if self.has_layer_norm:
            x = self.fc1(x)
        x = self.fc2(x)
        return x
    
    def evaluate(self, X, attention, y, criterion, device, other_class=0, batch_size = 32):
        with torch.no_grad():
            outputs = torch.tensor([], device=device)
            for idx in range(math.ceil(len(X)/batch_size)):
                inputs_0 = X[idx*batch_size:min(len(X), (idx+1)*batch_size)].to(device)
                input_attention = attention[idx*batch_size:min(len(attention), (idx+1)*batch_size)].to(device)
                outputs = torch.cat((outputs, self(inputs_0, input_attention)), 0)
        _, predicted = torch.max(outputs.data, 1)
        total = y.size(0)
        correct = (predicted == y.to(device)).sum().item()
        accuracy = correct/total
        loss = criterion(outputs, y.to(device)).item()
        if self.no_classes==37 and other_class==0:
            t = 0
            for i in range(18):
                t+=f1_score(y.cpu(), predicted.cpu(), average='micro', labels=[2*i+1,2*i+2])
            f1 = t/18
        else:
            print(f'Evaluation should be added manually for {self.no_classes} classes and other class #{other_class}')
            return 0, 0, 0, np.array(predicted.cpu())
        return accuracy, f1, loss, np.array(predicted.cpu())

model = Model().to(device)

### Input/Tokenization Details

def read_kbp_format(fp):
    with open(fp) as f:
        raw_lines = f.read().splitlines()
    
    X = []
    y = []
    for i in range(len(raw_lines)//4):
        x = raw_lines[4*i].split('\t')[1][1:-1]
        x = x.replace('<e1>', '<e1> ').replace('</e1>', ' </e1>').replace('<e2>', '<e2> ').replace('</e2>', ' </e2>')
        y_temp = raw_lines[4*i+1].strip()
        X.append(x)
        y.append(y_temp)

    y_cats = []
    for y_el in y:
        y_cats.append(categories.index(y_el))

    return X, torch.tensor(y_cats)

def to_id(text, representation = 'marker'):
    new_text = []
    for word in text.split():
        if word.startswith('http'):
            continue
        elif word.startswith('www'):
            continue
        elif word.startswith('**********'):
            continue
        elif word.startswith('-------'):
            continue
        new_text.append(word)
    text = ' '.join(new_text)
    if representation=='marker':
        if text.index('<e1>')<text.index('<e2>'):
            fc = 'e1'
            sc = 'e2'
            be1 = 1
            le1 = 2
            be2 = 3
            le2 = 4
        else:
            fc = 'e2'
            sc = 'e1'
            be1 = 3
            le1 = 4
            be2 = 1
            le2 = 2
        initial = tokenizer.encode(text[:text.index(f'<{fc}>')].strip(), add_special_tokens=False)
        e1 = tokenizer.encode(text[text.index(f'<{fc}>')+4:text.index(f'</{fc}>')].strip(), add_special_tokens=False)
        middle = tokenizer.encode(text[text.index(f'</{fc}>')+5:text.index(f'<{sc}>')].strip(), add_special_tokens=False)
        e2 = tokenizer.encode(text[text.index(f'<{sc}>')+4:text.index(f'</{sc}>')].strip(), add_special_tokens=False)
        final = tokenizer.encode(text[text.index(f'</{sc}>')+5:].strip(), add_special_tokens=False)
        return torch.tensor([101]+initial+[be1]+e1+[le1]+middle+[be2]+e2+[le2]+final+[102])

def feature_extraction(fp):
    X, y = read_kbp_format(fp)
    features = []
    attention_masks = []
    for sentence in tqdm(X):
        input_ids = to_id(sentence, representation='marker')        
        if len(input_ids)>=max_seq_length:
            input_ids = input_ids[:max_seq_length-1]
        attention_mask = torch.cat((torch.tensor([1.0]*(len(input_ids))), torch.tensor([0.0]*(max_seq_length-len(input_ids)))), 0)
        input_ids = torch.cat((input_ids, torch.tensor([0]*(max_seq_length-len(input_ids)))), 0)
        attention_masks.append(attention_mask)
        features.append(input_ids)
    return torch.stack(features), torch.stack(attention_masks), y

categories = []
with open("train.txt") as f:
    raw_lines = f.read().splitlines()

for i in range(len(raw_lines)//4):
    categories.append(raw_lines[4*i+1].strip())

categories = sorted(list(set(categories)))

X_train_feat, X_train_attention, y_train = feature_extraction("train.txt")
X_dev_feat, X_dev_attention, y_dev = feature_extraction("dev.txt")
X_test_feat, X_test_attention, y_test = feature_extraction("test.txt")

X_relx_de_feat, X_relx_de_att, y_relx_de = feature_extraction("RELX_de.txt")
X_relx_en_feat, X_relx_en_att, y_relx_en = feature_extraction("RELX_en.txt")
X_relx_es_feat, X_relx_es_att, y_relx_es = feature_extraction("RELX_es.txt")
X_relx_fr_feat, X_relx_fr_att, y_relx_fr = feature_extraction("RELX_fr.txt")
X_relx_tr_feat, X_relx_tr_att, y_relx_tr = feature_extraction("RELX_tr.txt")

### Setting the hyperparameters

counts = np.array([Counter(y_train.numpy())[i] for i in range(37)])
weights = [1]
for i in range(1, 37, 2):
    weights.append(counts[i]/(counts[i]+counts[i+1]))
    weights.append(counts[i+1]/(counts[i]+counts[i+1]))
weights = torch.Tensor(weights).to(device)

lr = 0.00003
wd = 0.1
criterion = nn.CrossEntropyLoss(weight=weights)
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": wd,
    },
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]

optimizer = AdamW(optimizer_grouped_parameters, lr=lr, weight_decay=wd)

### Training & Saving the best model

batch_size = 16
best_val_f1 = 0
accumulation_steps = 4
for epoch in range(10):
    running_loss = 0.0
    total_loss = 0.0
    total = 0
    correct = 0
    indices = np.arange(len(X_train_feat))
    np.random.shuffle(indices)
    train_outputs = torch.LongTensor([]).to(device)
    for idx in range(math.ceil(len(X_train_feat)/batch_size)):
        inputs_0 = X_train_feat[indices[idx*batch_size:min(len(X_train_feat), (idx+1)*batch_size)]].to(device)
        input_attention = X_train_attention[indices[idx*batch_size:min(len(X_train_attention), (idx+1)*batch_size)]].to(device)
        labels = y_train[indices[idx*batch_size:min(len(y_train), (idx+1)*batch_size)]].to(device)

        outputs = model(inputs_0, input_attention)
        loss = criterion(outputs, labels) / accumulation_steps 
        loss.backward()
        
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total+= len(labels)
        train_outputs = torch.cat((train_outputs, predicted), 0)
        if (idx+1)%accumulation_steps==0:             
            optimizer.step()                            
            optimizer.zero_grad()


        # print statistics
        running_loss += loss.item()
        if (idx+1) % accumulation_steps == 0:   # print every 2000 mini-batches
            print('[%d_%d, %5d/%d] loss: %.3f accuracy: %.3f' %
                  (epoch + 1, (idx+1) % accumulation_steps, idx + 1, len(X_train_feat)//batch_size, running_loss, correct/total))
            total_loss += running_loss
            running_loss = 0.0
      
    train_acc = correct/total
    t = 0
    for i in range(18):
        t+=f1_score(y_train[indices].cpu(), train_outputs.cpu(), average='micro', labels=[2*i+1,2*i+2])
    train_f1 = t/18

    val_acc, val_f1, val_loss, _ = model.evaluate(X_dev_feat, X_dev_attention, y_dev, criterion, device, other_class=0)
    test_acc, test_f1, test_loss, _ = model.evaluate(X_test_feat, X_test_attention, y_test, criterion, device, other_class=0)



    de_acc, de_f1, de_loss, _ = model.evaluate(X_relx_de_feat, X_relx_de_att, y_relx_de, criterion, device, other_class=0)
    en_acc, en_f1, en_loss, _ = model.evaluate(X_relx_en_feat, X_relx_en_att, y_relx_en, criterion, device, other_class=0)
    es_acc, es_f1, es_loss, _ = model.evaluate(X_relx_es_feat, X_relx_es_att, y_relx_es, criterion, device, other_class=0)
    fr_acc, fr_f1, fr_loss, _ = model.evaluate(X_relx_fr_feat, X_relx_fr_att, y_relx_fr, criterion, device, other_class=0)
    tr_acc, tr_f1, tr_loss, _ = model.evaluate(X_relx_tr_feat, X_relx_tr_att, y_relx_tr, criterion, device, other_class=0)

    if val_f1>best_val_f1:
        now = datetime.now()
        print(f'{now}  :  {val_f1} is higher than the best({best_val_f1}). Saving the model at ../Models/KBP37/bert_multilingual_adam_finetune_kbp37_{epoch+1}_{test_f1}_sigmoid_long2.pt')
        best_val_f1 = val_f1
        torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss
                }, f'bert_{epoch+1}_{test_f1}.pt')

        
    print('Epoch: ',epoch+1)
    print(f'Training Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}\nTraining accuracy:{train_acc:.4f}, Training F1:{train_f1:.4f}, Validation accuracy:{val_acc:.4f}, Validation F1:{val_f1:.4f}, Test accuracy:{test_acc:.4f}, Test F1:{test_f1:.4f}')
    print(f"German F1: {de_f1:.4f}, English F1: {en_f1:.4f}, Spanish F1: {es_f1:.4f}, French F1: {fr_f1:.4f}, Turkish F1: {tr_f1:.4f}")
