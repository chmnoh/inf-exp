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
    dirz = [ '../models/ftout/'+n for n in os.listdir('../models/ftout') ]
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
    for text in f.read().decode('utf-8').split('\n'):
        text = text.strip()
        if len(text) == 0:
            continue
        inputs = tokenizer(text, return_tensors="pt")
        inputs.to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            #logits_cpu = logits.cpu()
            #logits_arr = logits_cpu.detach().numpy()
            probs = F.softmax(logits, dim=1)
        #print("len(logits)={} {} len(probs)={}".format(len(logits), type(logits_cpu), len(probs)))
        predicted_class_id = logits.argmax().item()
        print('"{}",{},{},(prob:{})'.format(text.replace('"','""'), predicted_class_id,
            model.config.id2label[predicted_class_id], probs[0,predicted_class_id]))
