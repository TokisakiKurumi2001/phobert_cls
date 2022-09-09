from phobert_cls import tokenizer, test_dataloader, PhoBERTForSeqClassifier, LitPhoBERForSeqClassifier
import torch

if __name__ == "__main__":
    # model = PhoBERTForSeqClassifier(num_classes=5)
    # checkpoint = torch.load('./checkpoint/epoch=19-step=100.ckpt')
    # print(checkpoint.keys())
    # model.load_state_dict(torch.load('./checkpoint/epoch=19-step=100.ckpt')['phobert_cls'])
    model = LitPhoBERForSeqClassifier(num_classes=5)
    model.load_from_checkpoint('./checkpoint/epoch=19-step=100.ckpt')
    model.eval()
    predicts = []
    print("Predicting")
    for batch in test_dataloader:
        sents = batch.pop('original_sents')
        logits = model.phobert_cls(batch)
        preds = torch.argmax(logits, dim=-1).tolist()
        predicts.extend([(sent, tokenizer.convertId2Label(pred)) for sent, pred in zip(sents, preds)])

    with open('predict.txt', 'w+') as file:
        for number, pred in predicts:
            file.write(f"{number}: {pred}\n")