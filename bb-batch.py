import sys, os, transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
from transformers.utils import logging
import numpy as np
import torch.nn.functional as F

if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
    print("usage: {} file [checkpoint]".format(sys.argv[0]))
    sys.exit(1)

logging.set_verbosity(transformers.logging.ERROR)

device = "cuda:0"
tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base", truncation=True)  # BertTokenizer

chkpt = ''
if len(sys.argv) > 2:
    chkpt = '../models/ftout/checkpoint-{}'.format(sys.argv[2])
else:
    dirz = [ 'ftout/'+n for n in os.listdir('ftout') ]
    llz = [ (p, int(os.stat(p).st_mtime)) for p in dirz ]
    modelz = sorted(llz, key=lambda t: t[1], reverse=True)
    print("using checkpoint: {} ...".format(modelz[0][0]))
    chkpt = modelz[0][0]
model = AutoModelForSequenceClassification.from_pretrained(chkpt, local_files_only=True)
model.to(device)

def softmax(logits):
    exps = np.exp(logits)
    return exps / np.sum(exps, axis=0)

with open(sys.argv[1],"rb") as f:
    sentences = []
    for text in f.read().decode('utf-8').split('\n'):
        text = text.strip()
        if len(text) == 0:
            continue
        sentences.append(text)
        # if len(sentences) > 1: break
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    inputs.to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        #logits_cpu = logits.cpu()
        #logits_arr = logits_cpu.detach().numpy()
        probs = F.softmax(logits, dim=1)
    #print("len(logits)={} {} len(probs)={}".format(len(logits), type(logits_cpu), len(probs)))
    probs, predictions = torch.max(probs, dim=1)
    probs_cpu = probs.cpu()
    predictions_cpu = predictions.cpu()
    probs_arr = probs_cpu.detach().numpy()
    predictions_arr = predictions_cpu.detach().numpy()
    preds = []
    for i, v in enumerate(predictions_arr):
        if probs_arr[i] < 0.7:
            preds.append(2)
        else:
            preds.append(v)
    print(probs_arr)
    print(predictions_arr.tolist())
    print(preds)
    print([model.config.id2label[p] for p in predictions_arr])
    # predicted_class_id = logits.argmax().item()
    # print('"{}",{},{},(prob:{})'.format(text.replace('"','""'), predicted_class_id,
    #     model.config.id2label[predicted_class_id], probs[0,predicted_class_id]))
