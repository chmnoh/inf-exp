from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch

device = "cuda:0"
tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")  # BertTokenizer

model = AutoModelForSequenceClassification.from_pretrained("../models/ftout/checkpoint-99624", local_files_only=True)
model.to(device)

text = "코로나2023 영향으로 마스크 착용이 감소하고 있다."
inputs = tokenizer(text, return_tensors="pt")
inputs.to(device)
with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
print(text)
print(predicted_class_id)
print(model.config.id2label[predicted_class_id])
