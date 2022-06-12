from torch.utils.data import DataLoader
from datasets import load_dataset,load_metric
from transformers import AdamW, AutoModel, get_scheduler,default_data_collator, AutoTokenizer, Trainer, TrainingArguments,BertTokenizer,AutoModelForSequenceClassification,DataCollatorWithPadding
from accelerate import Accelerator
from tqdm.auto import tqdm
from argparse import ArgumentParser
import numpy as np
import torch
import time
import gc
import psutil
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = ArgumentParser()
parser.add_argument('-m', '--model', default='bert-base-cased')
parser.add_argument('-b', '--batch_size', default=16)
parser.add_argument('-c', '--chunk_size', default=32)
parser.add_argument('-e', '--epochs', default=4)
#parser.add_argument('-train', '--trainingData', default='/home/trawat2/LearningProject/Iwouldrather/data/train.csv')
#parser.add_argument('-val', '--validationData', default='/home/trawat2/LearningProject/Iwouldrather/data/valid.csv')
parser.add_argument('-s', '--saveDir', default='/home/trawat2/LearningProject/Iwouldrather/test-boolq/')
args = parser.parse_args()


model_ckpt = str(args.model)
batch_size = int(args.batch_size)
chunk_size = int(args.chunk_size)
num_epochs = int(args.epochs)
#trainPath = str(args.trainingData)
#validPath = str(args.validationData)
savePath = str(args.saveDir)
learning_rate = 3e-5
benchmark="super_glue"
data="boolq"

if not os.path.exists(savePath):
    os.makedirs(savePath)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
accelerator = Accelerator()



def preprocessedDataset():
    #boolq_dataset = load_dataset("super_glue","boolq")
    def prepare_train_features(examples):
        tokenized_examples = tokenizer(
            examples["question"],
            examples["passage"],
            truncation=True
        )
        return tokenized_examples
    dataset = load_dataset(benchmark, data)

    dataset = dataset.map(prepare_train_features, batched=True)
    dataset = dataset.remove_columns(['question', 'passage', 'idx'])
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(dataset['train'], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    eval_loader = DataLoader(dataset['validation'], batch_size=batch_size, collate_fn=data_collator)
    return  dataset , train_loader, eval_loader
    
def computeMetrics(eval_pred):
    accuracy_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}
    
def createTraingArguments():
    
    args = TrainingArguments(
    savePath,
    evaluation_strategy = "epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,#BATCH_SIZE,
    per_device_eval_batch_size=batch_size,#BATCH_SIZE,
    num_train_epochs=num_epochs, # Set num_train_epochs to 1 as test
    weight_decay=0.01,
    )
 
    return args
    
def getModel():
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2)
    model.to(device)
    return model

def training():
    dataset,train_loader, eval_loader = preprocessedDataset()
    
    model=getModel()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Clearing cache
    gc.collect()
    torch.cuda.empty_cache()
    
    train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
        train_loader, eval_loader, model, optimizer
    )
    
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    print(" Training:::: ")
    
    '''
    model.train()
    #progressBar = tqdm(range(num_training_steps))
   
    
    for epoch in range(num_epochs):
        print(f'\nEpoch: {epoch + 1} of {num_epochs}')
        for batch in train_dataloader:
            # batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            # loss.backward()
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
           #progressBar.update(1)
           
    
    print(" Evaluation:::: ")
    metric_acc = load_metric("glue", "mnli")
    metric_f1 = load_metric("f1")
    model.eval()
    
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        metric_acc.add_batch(predictions=accelerator.gather(predictions),references=accelerator.gather(batch["labels"]))
        metric_f1.add_batch(predictions=accelerator.gather(predictions), references=accelerator.gather(batch["labels"]))
        
    acc_score = metric_acc.compute()
    f1_score = metric_f1.compute(average="weighted")
    print('accuracy ::::: ', acc_score)
    print('f1_score ::::: ', f1_score)
    
    '''
    
    args=createTraingArguments()
    
    trainer = Trainer(
    model,
    args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    compute_metrics=computeMetrics,
    # data_collator=data_collator, # I had to comment this because it was not working with the default collator
    tokenizer=tokenizer)
    
    trainer.train()
    
    

if __name__ == '__main__':
    training()
    
    

