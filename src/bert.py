import torch
from transformers import BertModel, AutoTokenizer
from argparse import ArgumentParser
import pandas as pd

parser = ArgumentParser()
parser.add_argument('-m', '--model', default='bert-base-cased')
args = parser.parse_args()

pretrained_model = str(args.model)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
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


if __name__ == '__main__':
    main()
