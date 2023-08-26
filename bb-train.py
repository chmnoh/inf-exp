from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding
import datasets
import evaluate
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")  # BertTokenizer
ds = datasets.load_dataset("../models/inflation-tone")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_ds = ds.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

label2id = {"incr-future":0, "incr-present":1, "neut":2, "decr-future":3, "decr-present":4}
id2label = {0:"incr-future", 1:"incr-present", 2:"neut", 3:"decr-future", 4:"decr-present"}

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "monologg/kobigbird-bert-base", num_labels=len(label2id.keys()), id2label=id2label, label2id=label2id
)
model.to("cuda:0")

lr = 1e-5
batch = 8
epochs = 1
decay = 0.001
training_args = TrainingArguments(
    output_dir="ftout",
    learning_rate=lr,
    per_device_train_batch_size=batch,
    per_device_eval_batch_size=batch,
    num_train_epochs=epochs,
    weight_decay=decay,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
print("lr={} batch={} epochs={} decay={}".format(
	lr, batch, epochs, decay))
