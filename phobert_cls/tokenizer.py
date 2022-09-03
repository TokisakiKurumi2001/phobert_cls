from transformers import PhobertTokenizer

class SeqClassifierTokenizer:
    def __init__(self):
        self.tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base")
        self.label2id = {
            'Usual': 0,
            'Social': 1,
            'Politics': 2,
            'Science': 3,
            'Education': 4
        }
        self.id2label = {v: k for k, v in self.label2id.items()}

    def tokenize(self, sentences, **kwargs):
        return self.tokenizer(sentences, **kwargs)

    def convertLabel2Id(self, label):
        return self.label2id[label]

    def convertId2Label(self, id):
        return self.id2label[id]

tokenizer = SeqClassifierTokenizer()