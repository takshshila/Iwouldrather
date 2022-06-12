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
parser.add_argument('-tb', '--train_batch_size', default=128)
parser.add_argument('-vb', '--eval_batch_size', default=256)
parser.add_argument('-c', '--chunk_size', default=32)
parser.add_argument('-e', '--epochs', default=15)
#parser.add_argument('-train', '--trainingData', default='/home/trawat2/LearningProject/Iwouldrather/data/train.csv')
#parser.add_argument('-val', '--validationData', default='/home/trawat2/LearningProject/Iwouldrather/data/valid.csv')
#parser.add_argument('-test', '--testingData', default='/home/trawat2/LearningProject/Iwouldrather/data/test.csv')
parser.add_argument('-s', '--saveDir', default='/home/trawat2/LearningProject/Iwouldrather/test-circa')
args = parser.parse_args()


pretrained_model = str(args.model)
train_batch_size = int(args.train_batch_size)
eval_batch_size = int(args.eval_batch_size)
chunk_size = int(args.chunk_size)
num_epochs = int(args.epochs)
#trainPath = str(args.trainingData)
#validPath = str(args.validationData)
#testPath = str(args.testingData)
savePath = str(args.saveDir)+'/'+str(num_epochs)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model, truncation=True, do_lower_case=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 2e-5

class BERT(torch.nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.l1 = BertModel.from_pretrained(pretrained_model)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 6)

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
        
        
def training(train_loader, valid_loader, savePath):
    print('Training !!! ')
    
    model = BERT()
    model.to(device)
    #model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    #gc.collect()
    #torch.cuda.empty_cache()
    # start time
    #startTime = time.time()
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
    
    
        model.eval()
        with torch.no_grad():
            print("Evaluation")
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
                
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    torch.save(model, savePath+'/model')
    
    
def testing(test_loader, savePath):
    
    metric_acc = load_metric("accuracy")
    metric_f1 = load_metric("f1")
    model = torch.load(savePath+'/model', map_location=device)
    y_pred = []
    y_target = []
    
    print("Testing Started!!!")
    
    model.eval()
    with torch.no_grad():
        for test_batch in test_loader:
            eval_targets = test_batch['labels'].to(device)
            input_ids = test_batch['input_ids'].to(device)
            attention_mask = test_batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            predictions = torch.argmax(outputs, dim=-1)
            y_pred.extend(predictions.cpu())
            y_target.extend(eval_targets.cpu())
             
            metric_acc.add_batch(predictions=predictions, references=eval_targets)
            metric_f1.add_batch(predictions=predictions, references=eval_targets)
            
    acc_score = metric_acc.compute()
    f1_score = metric_f1.compute(average="weighted")
    print('Testing Accuracy: {}, F1 Score: {}'.format(acc_score, f1_score))
    print(metrics.classification_report(y_target, y_pred, digits=3,target_names=['Yes', 'No', 'In the middle, neither yes nor no', 'Yes, subject to some conditions','Other','NA']))
    #print('accuracy ::::: ', acc_score)
    #print('f1_score ::::: ', f1_score)

def stringToLabel(row):
    if row['goldstandard2']=='Yes':
        row['rlabels'] = 0
    elif row['goldstandard2']=='No':
        row['rlabels'] = 1
    elif row['goldstandard2']=='In the middle, neither yes nor no':
        row['rlabels'] = 2
    elif row['goldstandard2']=='Yes, subject to some conditions':
        row['rlabels'] = 3
    elif row['goldstandard2']=='Other':
        row['rlabels'] = 4
    else:
        row['rlabels'] = 5
        
    return row
    
def datasetCreationAnswerOnly():
    #df = pd.read_csv('/home/trawat2/LearningProject/circa_Aonly.csv', sep=',')
    #relaxedDF = df[['answer-Y','rlabels']]
    #relaxedDF=relaxedDF.rename(columns={"rlabels": "labels"})
    
    #train=relaxedDF.sample(frac=0.6,random_state=200) #random state is a seed value
    #val_test=relaxedDF.drop(train.index)
    
    #val = val_test.sample(frac=0.5,random_state=200)
    #test=val_test.drop(val.index)
    train=pd.read_csv('/home/trawat2/LearningProject/circa_train.csv', sep=',')
    val=pd.read_csv('/home/trawat2/LearningProject/circa_val.csv', sep=',')
    test=pd.read_csv('/home/trawat2/LearningProject/circa_test.csv', sep=',')
    
    train_encodings = tokenizer(train['answer-Y'].tolist(), truncation=True, padding=True)
    val_encodings = tokenizer(val['answer-Y'].tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(test['answer-Y'].tolist(), truncation=True, padding=True)
    
    train_dataset = Dataset(train_encodings, list(train['rlabels']))
    valid_dataset = Dataset(val_encodings, list(val['rlabels']))
    test_dataset = Dataset(test_encodings, list(test['rlabels']))
    
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
    
    return train_loader,valid_loader,test_loader

def datasetCreationQuestionOnly():
    train=pd.read_csv('/home/trawat2/LearningProject/circa_train.csv', sep=',')
    val=pd.read_csv('/home/trawat2/LearningProject/circa_val.csv', sep=',')
    test=pd.read_csv('/home/trawat2/LearningProject/circa_test.csv', sep=',')
    
    train_encodings = tokenizer(train['question-X'].tolist(), truncation=True, padding=True)
    val_encodings = tokenizer(val['question-X'].tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(test['question-X'].tolist(), truncation=True, padding=True)
    
    train_dataset = Dataset(train_encodings, list(train['rlabels']))
    valid_dataset = Dataset(val_encodings, list(val['rlabels']))
    test_dataset = Dataset(test_encodings, list(test['rlabels']))
    
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
    
    return train_loader,valid_loader,test_loader
    
def datasetCreationQuestionAnswerPair():
    
    train=pd.read_csv('/home/trawat2/LearningProject/circa_train.csv', sep=',')
    val=pd.read_csv('/home/trawat2/LearningProject/circa_val.csv', sep=',')
    test=pd.read_csv('/home/trawat2/LearningProject/circa_test.csv', sep=',')
    
    train_encodings = tokenizer(train['question-X'].tolist(),train['answer-Y'].tolist(), truncation=True, padding=True)
    val_encodings = tokenizer(val['question-X'].tolist(),val['answer-Y'].tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(test['question-X'].tolist(),test['answer-Y'].tolist(), truncation=True, padding=True)
    
    train_dataset = Dataset(train_encodings, list(train['rlabels']))
    valid_dataset = Dataset(val_encodings, list(val['rlabels']))
    test_dataset = Dataset(test_encodings, list(test['rlabels']))
    
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
    
    return train_loader,valid_loader,test_loader

    
    
if __name__ == '__main__':
    print("Answer Only")
    train_loader, valid_loader, test_loader = datasetCreationAnswerOnly()
    training(train_loader, valid_loader, savePath+'/AnswerOnly')
    torch.save(tokenizer, savePath+'/tokenizer')
    testing(test_loader, savePath+'/AnswerOnly')
    
    print("Question Only")
    train_loader, valid_loader, test_loader = datasetCreationQuestionOnly()
    training(train_loader, valid_loader, savePath+"/QuestionOnly")
    testing(test_loader, savePath+"/QuestionOnly")
    torch.save(tokenizer, savePath+'/tokenizer')
    
    print("Question and Answer Pair")
    train_loader, valid_loader, test_loader = datasetCreationQuestionAnswerPair()
    training(train_loader, valid_loader, savePath+"/QuestionAnswer")
    testing(test_loader, savePath+"/QuestionAnswer")
    torch.save(tokenizer, savePath+'/tokenizer')

    
    
    
