from phobert_cls import tokenizer, test_dataloader, LitPhoBERForSeqClassifier
import torch
import pandas as pd

if __name__ == "__main__":
    model = LitPhoBERForSeqClassifier.load_from_checkpoint('./checkpoint/epoch=7-step=40.ckpt')
    model.eval()
    model.freeze()
    origin_sents = []
    pred_labels = []
    print("Predicting")
    for batch in test_dataloader:
        sents = batch.pop('original_sents')
        logits = model.phobert_cls(batch)
        preds = torch.argmax(logits, dim=-1).tolist()
        origin_sents.extend(sents)
        pred_labels.extend([tokenizer.convertId2Label(pred) for pred in preds])

    df = pd.DataFrame({'vi_wseg': origin_sents, 'labels': pred_labels})
    df.to_csv('predict.csv', index=False)