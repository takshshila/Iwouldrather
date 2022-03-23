import torch
from torch.utils.data import DataLoader
from transformers import BertModel, AutoTokenizer
from argparse import ArgumentParser
import pandas as pd

parser = ArgumentParser()
parser.add_argument('-m', '--model', default='bert-base-cased')
parser.add_argument('-b', '--batch_size', default=32)
args = parser.parse_args()

pretrained_model = str(args.model)
batch = int(args.batch_size)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def main():
    train = pd.read_csv()
    valid = pd.read_csv()
    test = pd.read_csv()

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, truncation=True, do_lower_case=True)

    train_encodings = tokenizer(list(train['text']), truncation=True, padding=True)
    valid_encodings = tokenizer(list(valid['text']), truncation=True, padding=True)
    test_encodings = tokenizer(list(test['text']), truncation=True, padding=True)

    train_dataset = Dataset(train_encodings, list(train['label'].astype(float)))
    valid_dataset = Dataset(valid_encodings, list(valid['label'].astype(float)))
    test_dataset = Dataset(test_encodings, list(test['label'].astype(float)))

    model = BERT()
    model.to(device)
    batch_size = batch
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


if __name__ == '__main__':
    main()
