from argparse import ArgumentParser
from datasets import load_dataset

parser = ArgumentParser()
parser.add_argument('-m', '--model', default='bert-base-cased')
parser.add_argument('-b', '--batch_size', default=32)
parser.add_argument('-e', '--epochs', default=10)

mnli_dataset = load_dataset('glue', 'mnli')


