import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertModel, AutoTokenizer
from datasets import load_metric
from argparse import ArgumentParser
import time
import gc
import psutil
import os
from sklearn import metrics

parser = ArgumentParser()
parser.add_argument('-m', '--model', default='bert-base-cased')
parser.add_argument('-b', '--batch_size', default=16)
parser.add_argument('-c', '--chunk_size', default=32)
parser.add_argument('-e', '--epochs', default=5)
parser.add_argument('-train', '--trainingData', default='/home/trawat2/LearningProject/Iwouldrather/data/train_mnli_new.csv')
parser.add_argument('-val', '--validationData', default='/home/trawat2/LearningProject/Iwouldrather/data/valid.csv')
parser.add_argument('-test', '--testingData', default='/home/trawat2/LearningProject/Iwouldrather/data/test_mnli_new.csv')
parser.add_argument('-s', '--saveDir', default='/home/trawat2/LearningProject/Iwouldrather/test-mnli')
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
out_path=savePath+'/MNLI_new'#+'/'+pretrained_model
learning_rate = 2e-5

class BERT(torch.nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.l1 = BertModel.from_pretrained(pretrained_model)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


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

def training(train_loader, valid_loader):
    print("Training Started!!!")
    #eval_every = int(len(train_loader) * 0.8)
    #global_step = 0
    
    model = BERT()
    model.to(device)
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
            #global_step += 1
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
    
    
    #print("---Training Time %s seconds ---" % (time.time() - startTime))
    #print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    
    
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
            y_pred.extend(predictions)
            y_target.extend(eval_targets)
            metric_acc.add_batch(predictions=predictions, references=eval_targets)
            metric_f1.add_batch(predictions=predictions, references=eval_targets)
            
        acc_score = metric_acc.compute()
        f1_score = metric_f1.compute(average="weighted")
  
        #print('{} Traning Loss: {:.4f} Evaluation Loss: {:.4f}'.format(epoch, train_loss, val_loss))
        print('Testing Accuracy: {}, F1 Score: {}'.format(acc_score, f1_score))
        #print('accuracy ::::: ', acc_score)
        #print('f1_score ::::: ', f1_score)
        
        print(metrics.confusion_matrix(y_target, y_pred))

        # Print the precision and recall, among other metrics
        print(metrics.classification_report(y_target, y_pred, digits=3))
    


def main():
    
    print('MNLI finetuning with new test data')
    train = pd.read_csv(trainPath, sep=',').dropna()
    valid = pd.read_csv(validPath, sep=',').dropna()
    test = pd.read_csv(testPath, sep=',').dropna()

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, truncation=True, do_lower_case=True)

    train_encodings = tokenizer(train['premise'].tolist(), train['hypothesis'].tolist(), truncation=True, padding=True)
    valid_encodings = tokenizer(valid['premise'].tolist(), valid['hypothesis'].tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(test['premise'].tolist(), test['hypothesis'].tolist(), truncation=True, padding=True)

    train_dataset = Dataset(train_encodings, list(train['label'].astype(float)))
    valid_dataset = Dataset(valid_encodings, list(valid['label'].astype(float)))
    test_dataset = Dataset(test_encodings, list(test['label'].astype(float)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    training(train_loader, valid_loader)
    
    
    torch.save(tokenizer, out_path+'/tokenizer')
    
    
    print("Testing MNLI finetuned model")
    
    # Clearing cache
    gc.collect()
    torch.cuda.empty_cache()
    
    tokenizer = torch.load(out_path+'/tokenizer', map_location=device)
    '''
    test = pd.read_csv(testPath, sep=',').dropna()
    test_encodings = tokenizer(test['premise'].tolist(), test['hypothesis'].tolist(), truncation=True, padding=True)
    test_dataset = Dataset(test_encodings, list(test['label'].astype(float)))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    '''
    
    testing(test_loader)
   
   
    
    '''
    # eval_every = int(len(train_loader) * 0.8)
    global_step = 0
    
    # Clearing cache
    gc.collect()
    torch.cuda.empty_cache()
    # start time
    startTime = time.time()

    for epoch in range(num_epochs):
        print(f'\nEpoch: {epoch + 1} of {num_epochs}')
        model.train()

        for batch in train_loader:
            targets = batch['labels'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            #outputs = torch.argmax(outputs, dim=-1)
            #targets=targets.to(dtype=torch.int64)
            
            #print(outputs, targets.long())
            loss = criterion(outputs, targets.long())
            #print("loss", loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            global_step += 1
            
            print('|', end='')

            # if global_step % eval_every == 0:
            #     model.eval()
            #
            #     with torch.no_grad():
            #         for valid_batch in valid_loader:
            #             targets = valid_batch['labels'].to(device)
            #             input_ids = valid_batch['input_ids'].to(device)
            #             attention_mask = valid_batch['attention_mask'].to(device)
            #             outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            #             outputs = outputs.reshape(-1)
            #             outputs.data = torch.tensor([1.0 if x.item() >= 0.5 else 0.0 for x in outputs.data]).to(device)
            #             criterion(outputs, targets)

    
    print("--- %s seconds ---" % (time.time() - startTime))
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    torch.save(model, savePath+'/model')
    torch.save(tokenizer, savePath+'/tokenizer')
    
    '''

if __name__ == '__main__':
    main()
