from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from phobert_cls import LitPhoBERForSeqClassifier, train_dataloader, valid_dataloader

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="phobert_cls")

    # model
    lit_phobert_cls = LitPhoBERForSeqClassifier(num_classes=5)

    # train model
    trainer = pl.Trainer(
        max_epochs=25, logger=wandb_logger, devices=2, accelerator="gpu", strategy="ddp",
        callbacks=[EarlyStopping(monitor="valid/acc_epoch", min_delta=0.00, patience=2, verbose=False, mode="max")]
    )
    trainer.fit(model=lit_phobert_cls, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
