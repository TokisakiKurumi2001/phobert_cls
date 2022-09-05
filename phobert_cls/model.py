import torch
import torch.nn as nn
from transformers import RobertaModel

class PhobertClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_classes):
        super(PhobertClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class PhoBERTForSeqClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PhoBERTForSeqClassifier, self).__init__()
        self.phobert = RobertaModel.from_pretrained("vinai/phobert-base")
        # for p in self.phobert.parameters():
        #     p.requires_grad = False
        self.classifier = PhobertClassificationHead(self.phobert.config, num_classes)
    
    def forward(self, inputs):
        outputs = self.phobert(**inputs)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)
        return logits