import torch
import torch.nn as nn
import pytorch_lightning as pl
from phobert_cls import PhoBERTForSeqClassifier

class LitPhoBERForSeqClassifier(pl.LightningModule):
    def __init__(self, num_classes):
        super(LitPhoBERForSeqClassifier, self).__init__()
        self.phobert_cls = PhoBERTForSeqClassifier(num_classes)
        self.num_classes = num_classes
        self.main_loss = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        logits = self.phobert_cls(batch)
        loss = self.main_loss(logits.view(-1, self.num_classes), labels.view(-1))
        self.log("train/loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        logits = self.phobert_cls(batch)
        loss = self.main_loss(logits.view(-1, self.num_classes), labels.view(-1))
        self.log("valid/loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        return optimizer