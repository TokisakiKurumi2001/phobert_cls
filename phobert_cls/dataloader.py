from phobert_cls import dataset, tokenizer
from torch.utils.data import DataLoader
from collections.abc import Mapping
import torch

def collate_fn(examples):
    if isinstance(examples, (list, tuple)) and isinstance(examples[0], Mapping):
        encoded_inputs = {key: [example[key] for example in examples] for key in examples[0].keys()}
    input = encoded_inputs['vi_wseg']
    tok = tokenizer.tokenize(input, padding='max_length', truncation=True, max_length=100, return_tensors='pt')
    
    labels = encoded_inputs['Categories']
    tok_labels = torch.tensor(labels)
    
    tok['labels'] = tok_labels
    return tok

def test_collate_fn(examples):
    if isinstance(examples, (list, tuple)) and isinstance(examples[0], Mapping):
        encoded_inputs = {key: [example[key] for example in examples] for key in examples[0].keys()}
    input = encoded_inputs['vi_wseg']
    tok = tokenizer.tokenize(input, padding='max_length', truncation=True, max_length=100, return_tensors='pt')
    tok['original_sents'] = encoded_inputs['vi_wseg']
    return tok

split_dataset = dataset['train'].class_encode_column('Categories')
split_dataset = split_dataset.train_test_split(test_size=0.1, stratify_by_column='Categories', seed=42)
split_dataset['valid'] = split_dataset.pop('test')
split_dataset['test'] = dataset['test']
split_dataset = split_dataset.remove_columns(['Unnamed: 0'])
# split_dataset['test'] = split_dataset['test'].remove_columns(['Unnamed: 0'])
# split_dataset['valid'] = split_dataset['valid'].remove_columns(['Unnamed: 0'])
train_dataloader = DataLoader(split_dataset['train'], batch_size=32, collate_fn=collate_fn, num_workers=24)
valid_dataloader = DataLoader(split_dataset['valid'], batch_size=32, collate_fn=collate_fn, num_workers=24)
test_dataloader = DataLoader(split_dataset['test'], batch_size=32, collate_fn=test_collate_fn, num_workers=24)