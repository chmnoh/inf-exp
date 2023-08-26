import os, sys

import json
import torch
from flask import Flask, redirect, render_template, request, url_for
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch.nn.functional as F

app = Flask(__name__)

device = "cuda:0"
tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")  # BertTokenizer
model = AutoModelForSequenceClassification.from_pretrained("../../models/final-ftout/checkpoint-5126", local_files_only=True)
model.to(device)

@app.route("/predict", methods=("GET", "POST"))
def predict():
    # print(request)
    # print("json: {} (type: {})".format(request.get_json(), type(request.get_json())))
    # return "json: {} (type: {})".format(request.get_json(), type(request.get_json()))
    inputs = tokenizer(request.get_json()['text'], return_tensors="pt")
    inputs.to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)
    predicted_class_id = logits.argmax().item()
    prob = probs[0,predicted_class_id].to(
            dtype=torch.double, 
            device=torch.device("cpu")).item()
    if prob < 0.70:
        predicted_class_id = 2  # neut
    # print(prob, type(prob))
    return {
        "class":model.config.id2label[predicted_class_id], 
        "prob":prob}

@app.route("/batchpredict", methods=("GET", "POST"))
def batchpredict():
    # print(request)
    # print("json: {} (type: {})".format(request.get_json(), type(request.get_json())))
    # return "json: {} (type: {})".format(request.get_json(), type(request.get_json()))
    inputs = tokenizer(request.get_json()['sentences'], padding=True, truncation=True, return_tensors="pt")
    inputs.to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)
    probs, predictions = torch.max(probs, dim=1)
    probs_cpu = probs.cpu()
    predictions_cpu = predictions.cpu()
    probs_arr = probs_cpu.detach().numpy()
    predictions_arr = predictions_cpu.detach().numpy()
    # print(probs_arr)
    # print(predictions_arr)
    # print([model.config.id2label[p] for p in predictions_arr])
    preds = []
    for i, v in enumerate(predictions_arr):
        if probs_arr[i] < 0.7:
            preds.append(2) # neut
        else:
            preds.append(v)
    classes = [model.config.id2label[p] for p in preds]
    # print(prob, type(prob))
    return {"classes":classes, "probs":probs_arr.tolist()}

