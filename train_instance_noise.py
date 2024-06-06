import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from sklearn import metrics
from sklearn import model_selection

from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup

import warnings

warnings.filterwarnings("ignore")


MAX_LEN = 192
BATCH_SIZE = 8
V_BATCH_SIZE = 8
TRAIN_PATH = './dataset/ag-news/train.csv'
EPOCHS = 5


#class to fetch a row from dataframe one by one and return dataset in well defined format

class prepare_dataset():
    def __init__(self, text, label):
        
        self.text = text
        self.label = label
        self.tokenizer = TOKENIZER
        self.max_len = MAX_LEN
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        
        text = self.text[idx]
        text = " ".join(text.split())
        label = self.label[idx]
        
        #Torkenize the text
        
        inputs = self.tokenizer.encode_plus(text , None,
                                           add_special_tokens=True,
                                           max_length = self.max_len,
                                           pad_to_max_length=True,
                                           truncation='longest_first',)
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        
        padding_length = self.max_len - len(ids)
        #pad the tokenized vectors so that each has the same length of 192
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        
        return {
            'ids' : torch.tensor(ids, dtype=torch.long),
            'masks' : torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(label, dtype=torch.float)
        }


#class to load the model
BERT_PATH = './models/bert-base-uncased'
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)

class BertBaseUncased(nn.Module):
    def __init__(self):
        super(BertBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(BERT_PATH)
        self.dropout = nn.Dropout(0.2)
        
#         for param in self.bert.parameters():
#             param.requires_grad = False

        unfreeze_layers = ['layer.10','layer.11','bert.pooler','out.']
        for name ,param in self.bert.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break

        # 4 output units in last layer since there are 4 classes 
        self.out = nn.Linear(768,4)                      
        
    def forward(self,ids, masks, token_type_ids):
        _, out = self.bert(ids, masks, token_type_ids, return_dict=False)
        out = self.dropout(out)
        out = self.out(out)
        
        return out      
    

#function to convert the integer classes to one hot encoded array
#example 2 -> [0, 0, 1, 0]
def ohe(df,target_col):
    encoded = pd.get_dummies(df.sort_values(by=[target_col])[target_col])
    df = df.join(encoded)
    return df

def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs,targets)

train_losses, val_losses = [], []
train_accu, val_accu = [], []

def train_loop(softmax_dataloader,dataloader, model, optimizer, device, scheduler):
    model.train()
    epoch_train_loss = 0
    epoch_train_acc = 0
    countertrain = 0

    for idx, batch in enumerate(dataloader):
        countertrain += 1
        ids = batch['ids']
        masks = batch['masks']
        token_type_ids = batch['token_type_ids']
        targets = batch['targets']

        ids = ids.to(device, dtype=torch.long)
        masks = masks.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()

        outputs = model(ids=ids, masks=masks, token_type_ids=token_type_ids)
        # print(outputs.shape)

        loss = loss_fn(outputs, targets)
        loss.backward()
        epoch_train_loss += loss.item()

        outputs = torch.argmax(outputs, axis=1)
        targets = torch.argmax(targets, axis=1)

        acc = metrics.accuracy_score(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy())
        epoch_train_acc += acc

        optimizer.step()

        scheduler.step()

        if idx % 50 == 0 and idx != 0:
            print(
                f'Batch: {idx}, train_loss: {epoch_train_loss / countertrain}, train_acc: {epoch_train_acc / countertrain}')

    train_losses.append(epoch_train_loss / countertrain)
    train_accu.append(epoch_train_acc / countertrain)

    model.eval()
    softmax_out = []
    for idx, batch in enumerate(softmax_dataloader):
        ids = batch['ids']
        masks = batch['masks']
        token_type_ids = batch['token_type_ids']

        ids = ids.to(device, dtype=torch.long)
        masks = masks.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        outputs = model(ids=ids, masks=masks, token_type_ids=token_type_ids)
        softmax_out.append(outputs)

    return epoch_train_acc / countertrain, epoch_train_loss / countertrain ,softmax_out


def eval_loop(dataloader, model, device):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    countertest = 0
    for idx, batch in enumerate(dataloader):
        countertest += 1
        ids = batch['ids']
        masks = batch['masks']
        token_type_ids = batch['token_type_ids']
        targets = batch['targets']

        ids = ids.to(device, dtype=torch.long)
        masks = masks.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        outputs = model(ids=ids, masks=masks, token_type_ids=token_type_ids)

        loss = loss_fn(outputs, targets)

        # get the index of the maximum value
        outputs = torch.argmax(outputs, axis=1)
        targets = torch.argmax(targets, axis=1)

        # calulate the accracy score
        acc = metrics.accuracy_score(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy())

        epoch_test_acc += acc
        epoch_test_loss += loss.item()

    val_losses.append(epoch_test_loss / countertest)
    val_accu.append(epoch_test_acc / countertest)
    final_acc = epoch_test_acc / countertest
    epoch_loss = epoch_test_loss / countertest
    return final_acc, epoch_loss


def softmax_loop(dataloader, model, device):
    model.eval()
    softmax_out = []
    for idx, batch in enumerate(dataloader):
        ids = batch['ids']
        masks = batch['masks']
        token_type_ids = batch['token_type_ids']
        targets = batch['targets']

        ids = ids.to(device, dtype=torch.long)
        masks = masks.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        outputs = model(ids=ids, masks=masks, token_type_ids=token_type_ids)
        softmax_out.append(outputs)

    return softmax_out


def train():
    
    df = pd.read_csv(TRAIN_PATH).fillna('None')
    #split the data into train and validation sets
    train, valid = model_selection.train_test_split(df, test_size = 0.15, random_state=42, stratify=df['Class Index'].values)
    
    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    
    #one hot encode the classes
    train= ohe(train, 'Class Index')
    valid = ohe(valid, 'Class Index')
    
    train_labels = train[train.columns[-4:]].values
    valid_labels = valid[valid.columns[-4:]].values
    
    
    train_data = prepare_dataset(text=train['Description'].values,
                                label=train_labels)
    
    valid_data = prepare_dataset(text=valid['Description'].values,
                                label=valid_labels)
    
    BATCH_SIZE = 16
    V_BATCH_SIZE = 16
    
    train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=BATCH_SIZE,num_workers=4,drop_last=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data,batch_size=V_BATCH_SIZE,num_workers=4,drop_last=True)
    
    
    
    device= torch.device('cuda:0')
    
    model = BertBaseUncased()
    model.to(device)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    
    num_train_steps = int(len(train_data)/BATCH_SIZE * EPOCHS)
    print(f'num_train_steps = {num_train_steps}')
    
#     lr = 1e-4 * xm.xrt_world_size()
    lr = 1e-4
    
    optimizer = AdamW(optimizer_parameters,lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    
    best_acc=0

    softmax_dataloader = torch.utils.data.DataLoader(train_data,batch_size=BATCH_SIZE,num_workers=4,drop_last=True)

    softmax_out_avg = np.zeros([len(train_data), 4]) 
    
    for epoch in range(EPOCHS):
        
        train_acc, train_loss,softmax_out = train_loop(softmax_dataloader,train_dataloader,model=model, optimizer=optimizer,scheduler=scheduler,device=device)
        softmax_out_avg += softmax_out
        # val_acc, val_loss = eval_loop(valid_dataloader, model, device)
        
        # print(f"EPOCH: {epoch} train_loss: {train_loss}  train_acc: {train_acc}  val_loss: {val_loss}  val_acc: {val_acc}")
        # if val_acc > best_acc:
        #     torch.save({'model':model.state_dict(), 'optimizer': optimizer.state_dict()},'./models/bert-base-uncased/best_model.bin')
        #     best_acc=val_acc
            
        # 在这里添加 softmax_out_avg   softmax_loader 是 train_dataloader 的不要shuffle版本
        # 不断累加   get_softmax_out(model, softmax_loader, device)
        #softmax_out_avg += softmax_loop(softmax_dataloader,model, device)

    # 这里计算softmax
    noise_rate = 0.2 
    label = np.array(train_labels)
    label_noisy_cand, label_noisy_prob = [], []
    for i in range(len(label)):
        pred = softmax_out_avg[i,:].copy()
        pred[label[i]] = -1
        # 迭代的取每一个样本，取得每一个样本的除去GT标签的最可能得标签和其对应的最高的置信度
        label_noisy_cand.append(np.argmax(pred))
        label_noisy_prob.append(np.max(pred))
        
    label_noisy = label.copy()
    # 取args.noise_rate的标签集合
    index = np.argsort(label_noisy_prob)[-int(noise_rate*len(label)):]
    # 标签翻转为噪声标签
    label_noisy[index] = np.array(label_noisy_cand)[index]

    save_pth = os.path.join('./data/CIFAR10/label_noisy', 'dependent'+str(noise_rate)+'.csv')
    pd.DataFrame.from_dict({'label':label,'label_noisy':label_noisy}).to_csv(save_pth, index=False)
    print('Noisy label data saved to ',save_pth)


torch.set_default_tensor_type('torch.FloatTensor')
a = train()