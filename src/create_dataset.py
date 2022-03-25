from argparse import ArgumentParser
from datasets import load_dataset
import pandas as pd
import json

parser = ArgumentParser()
parser.add_argument('-d', '--data', nargs='+', default=['glue', 'mnli'])
args = parser.parse_args()

dataset = load_dataset(args.data[0], args.data[1])
config = {}

df = pd.DataFrame()
df['premise'] = list(dataset['train']['premise'])
df['hypothesis'] = list(dataset['train']['hypothesis'])
df['label'] = list(dataset['train']['label'])
df.to_csv('../data/train.csv', index=False)
print('wrote:', len(df), 'rows to train.csv')
config['train_length'] = len(df)

df = pd.DataFrame()
df['premise'] = list(dataset['validation_matched']['premise'])
df['hypothesis'] = list(dataset['validation_matched']['hypothesis'])
df['label'] = list(dataset['validation_matched']['label'])
print('wrote:', len(df), 'rows to valid.csv')
df.to_csv('../data/valid.csv', index=False)
config['valid_length'] = len(df)

df = pd.DataFrame()
df['premise'] = list(dataset['test_matched']['premise'])
df['hypothesis'] = list(dataset['test_matched']['hypothesis'])
df['label'] = list(dataset['test_matched']['label'])
print('wrote:', len(df), 'rows to test.csv')
df.to_csv('../data/test.csv', index=False)
config['test_length'] = len(df)

with open('../data/params.json', 'w') as f:
    json.dump(config, f)
