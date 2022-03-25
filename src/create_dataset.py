from argparse import ArgumentParser
from datasets import load_dataset
import pandas as pd

parser = ArgumentParser()
parser.add_argument('-d', '--data', nargs='+', default=['glue', 'mnli'])
args = parser.parse_args()

dataset = load_dataset(args.data[0], args.data[1])

df = pd.DataFrame()
df['premise'] = list(dataset['train']['premise'])
df['hypothesis'] = list(dataset['train']['hypothesis'])
df['label'] = list(dataset['train']['label'])
df.to_csv('../data/train.csv')
print('wrote:', len(df), 'rows to train.csv')

df = pd.DataFrame()
df['premise'] = list(dataset['validation_matched']['premise'])
df['hypothesis'] = list(dataset['validation_matched']['hypothesis'])
df['label'] = list(dataset['validation_matched']['label'])
print('wrote:', len(df), 'rows to valid.csv')
df.to_csv('../data/valid.csv')

df = pd.DataFrame()
df['premise'] = list(dataset['test_matched']['premise'])
df['hypothesis'] = list(dataset['test_matched']['hypothesis'])
df['label'] = list(dataset['test_matched']['label'])
print('wrote:', len(df), 'rows to test.csv')
df.to_csv('../data/test.csv')
