from datasets import load_dataset, DatasetDict
dataset = load_dataset('csv', data_files=["data/train_wseg.csv"])
dataset['test'] = load_dataset('csv', data_files=["data/test_wseg.csv"]).pop('train')