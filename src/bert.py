import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertModel, AutoTokenizer
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-m', '--model', default='bert-base-cased')
parser.add_argument('-b', '--batch_size', default=32)
parser.add_argument('-c', '--chunk_size', default=32)
parser.add_argument('-e', '--epochs', default=10)
args = parser.parse_args()

pretrained_model = str(args.model)
batch_size = int(args.batch_size)
chunk_size = int(args.chunk_size)
num_epochs = int(args.epochs)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(pretrained_model, truncation=True, do_lower_case=True)


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
    def __init__(self, csv_path, chunkSize):
        self.chunksize = chunkSize
        self.reader = pd.read_csv(csv_path, sep=',', chunksize=self.chunksize, header=None, iterator=True)

    def __getitem__(self, idx):
        data = self.reader.get_chunk(self.chunksize)
        encodings = tokenizer(list(data.values[0]), list(data.values[1]), truncation=True, padding=True)
        item = {key: torch.tensor(val[idx]) for key, val in encodings.items()}
        item['labels'] = torch.tensor(data.values[2][idx]).to(device)
        return item

    def __len__(self):
        return self.chunksize


def main():

    model = BERT()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)

    train_loader = DataLoader(Dataset('../data/train.csv', chunk_size), batch_size=batch_size, shuffle=False)
    # valid_loader = DataLoader(Dataset('../data/valid.csv', chunk_size), batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(Dataset('../data/test.csv', chunk_size), batch_size=batch_size, shuffle=False)

    # eval_every = int(len(train_loader) * 0.8)
    global_step = 0

    for epoch in range(num_epochs):
        print(f'\nEpoch: {epoch + 1} of {num_epochs}')
        model.train()

        for batch in train_loader:
            targets = batch['labels'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = outputs.reshape(-1)
            outputs.data = torch.tensor([1.0 if x.item() >= 0.5 else 0.0 for x in outputs.data]).to(device)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

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


if __name__ == '__main__':
    main()
