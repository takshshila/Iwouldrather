import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertModel,AdamW,BertForSequenceClassification, get_scheduler,default_data_collator, AutoTokenizer,AutoModelForSequenceClassification,DataCollatorWithPadding
from accelerate import Accelerator
from datasets import load_metric
from argparse import ArgumentParser
import time
import gc
import psutil
import os
from sklearn import metrics

parser = ArgumentParser()
parser.add_argument('-m', '--model', default='bert-base-cased')
parser.add_argument('-b', '--batch_size', default=10)
parser.add_argument('-c', '--chunk_size', default=32)
parser.add_argument('-e', '--epochs', default=5)
parser.add_argument('-train', '--trainingData', default='/home/trawat2/LearningProject/Iwouldrather/data/boolq/train.csv')
parser.add_argument('-val', '--validationData', default='/home/trawat2/LearningProject/Iwouldrather/data/boolq/valid.csv')
parser.add_argument('-test', '--testingData', default='/home/trawat2/LearningProject/Iwouldrather/data/boolq/test.csv')
parser.add_argument('-s', '--saveDir', default='/home/trawat2/LearningProject/Iwouldrather/test-boolq_new')
args = parser.parse_args()


pretrained_model = str(args.model)
batch_size = int(args.batch_size)
chunk_size = int(args.chunk_size)
num_epochs = int(args.epochs)
trainPath = str(args.trainingData)
validPath = str(args.validationData)
testPath = str(args.testingData)
savePath = str(args.saveDir)



device = 'cuda' if torch.cuda.is_available() else 'cpu'
out_path=savePath
learning_rate = 3e-5


class BERT(torch.nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.l1 = BertModel.from_pretrained(pretrained_model)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
        
def getModel(model_ckpt):
    model = BertForSequenceClassification.from_pretrained(model_ckpt, num_labels=2)
    model.to(device)
    return model
    
def getTokenizer(model_ckpt):
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    return tokenizer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).to(device)
        return item

    def __len__(self):
        return len(self.labels)

def trainingONModel(train_loader, valid_loader):
    
    accelerator = Accelerator()
    model = getModel(pretrained_model)
    tokenizer = getTokenizer(pretrained_model)
    
    optimizer = AdamW(model.parameters(), lr=lr)

    train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
        train_loader, valid_loader, model, optimizer)
    
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    
    metric_acc = load_metric("accuracy")
    metric_f1 = load_metric("f1")
        
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
        model.eval()
        
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                metric_acc.add_batch(predictions=accelerator.gather(predictions),
                                     references=accelerator.gather(batch["labels"]))
                metric_f1.add_batch(predictions=accelerator.gather(predictions), references=accelerator.gather(batch["labels"]))

        acc_score = metric_acc.compute()
        f1_score = metric_f1.compute(average="weighted")
        print('{} Accuracy: {}, F1 Score: {}'.format(epoch, acc_score, f1_score))
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    torch.save(model, out_path+'/model')
    
    
def training(train_loader, valid_loader):
    model = BERT()
    model.to(device)
    #model = getModel(pretrained_model)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Clearing cache
    gc.collect()
    torch.cuda.empty_cache()
    
    # start time
    startTime = time.time()
    metric_acc = load_metric("accuracy")
    metric_f1 = load_metric("f1")
    
 
            
            
    for epoch in range(num_epochs):
        print(f'\nEpoch: {epoch + 1} of {num_epochs}')
        model.train()
        for batch in train_loader:
            targets = batch['labels'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            train_loss = criterion(outputs, targets.long())
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            print('|', end='')
        print()   
        model.eval()
        with torch.no_grad():
            for valid_batch in valid_loader:
                eval_targets = valid_batch['labels'].to(device)
                input_ids = valid_batch['input_ids'].to(device)
                attention_mask = valid_batch['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs, dim=-1)
                
                metric_acc.add_batch(predictions=predictions, references=eval_targets)
                metric_f1.add_batch(predictions=predictions, references=eval_targets)
            
            acc_score = metric_acc.compute()
            f1_score = metric_f1.compute(average="weighted")
            print('{} Accuracy: {}, F1 Score: {}'.format(epoch, acc_score, f1_score))
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    torch.save(model, out_path+'/model')
    
def testing(test_loader):
    # Clearing cache
    gc.collect()
    torch.cuda.empty_cache()
    y_pred = []
    y_target = []
    
    metric_acc = load_metric("accuracy")
    metric_f1 = load_metric("f1")
    model = torch.load(out_path+'/model', map_location=device)
    
    print("Testing Started!!!")
    
    model.eval()
    with torch.no_grad():
        for test_batch in test_loader:
            eval_targets = test_batch['labels'].to(device)
            input_ids = test_batch['input_ids'].to(device)
            attention_mask = test_batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            predictions = torch.argmax(outputs, dim=-1)
            
            #print(type(predictions), type(eval_targets))
            y_pred.extend(predictions.cpu())
            y_target.extend(eval_targets.cpu())
            
            
            metric_acc.add_batch(predictions=predictions, references=eval_targets)
            metric_f1.add_batch(predictions=predictions, references=eval_targets)
            
        acc_score = metric_acc.compute()
        f1_score = metric_f1.compute(average="weighted")
  
        print('Testing Accuracy: {}, F1 Score: {}'.format(acc_score, f1_score))
        

        print(metrics.classification_report(y_target, y_pred, digits=3))
    


def main():
    '''
    print('BOOLQ finetuning with new test data')
    
    train = pd.read_csv(trainPath, sep=',').dropna()
    valid = pd.read_csv(validPath, sep=',').dropna()
    

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, truncation=True, do_lower_case=True)

    train_encodings = tokenizer(train['question'].tolist(), train['passage'].tolist(), truncation=True, padding=True)
    valid_encodings = tokenizer(valid['question'].tolist(), valid['passage'].tolist(), truncation=True, padding=True)
    

    train_dataset = Dataset(train_encodings, list(train['label'].astype(float)))
    valid_dataset = Dataset(valid_encodings, list(valid['label'].astype(float)))
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    
    training(train_loader, valid_loader)
    
    
    torch.save(tokenizer, out_path+'/tokenizer')
    '''
    
    print("Testing BOOLQ finetuned model")
    tokenizer = torch.load(out_path+'/tokenizer')
    # Clearing cache
    gc.collect()
    torch.cuda.empty_cache()
    test = pd.read_csv(testPath, sep=',').dropna()
    print(test.shape)
    test_encodings = tokenizer(test['question'].tolist(), test['passage'].tolist(), truncation=True, padding=True)
    test_dataset = Dataset(test_encodings, list(test['label'].astype(float)))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    testing(test_loader)


if __name__ == '__main__':
    main()
