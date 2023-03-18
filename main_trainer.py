from timeit import default_timer as timer
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertConfig

from transformers import Trainer, TrainingArguments

from model import ModelClass
# from trainer import train
# from dataset import datasets

# Set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

with open("config.yaml", "r", encoding='utf_8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = config["NUM_EPOCHS"]
LR = float(config["LR"])
TRAIN_BATCH = config["TRAIN_BATCH"]
VAL_BATCH = config["VAL_BATCH"]
NUM_WORKERS = config["NUM_WORKERS"]


MAX_LEN = config["MAX_LEN"]
MODEL = config["MODEL"]
NUM_LABELS = config["NUM_LABELS"]

# Tokenizer

# Dataset


from datasets import load_dataset

emotions = load_dataset("emotion")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def tokenize(batch):
    return tokenizer(batch["text"], padding=False, truncation=True)

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

# Set up model
model_class = ModelClass(model_name=MODEL, num_labels=NUM_LABELS, len_train_dl=1000, lr=LR, epochs=NUM_EPOCHS)

model = model_class.model
# optimizer, scheduler = model_class.opt_sch()

# Start the timer

start_time = timer()



def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    # return {"accuracy": acc, "f1": f1}
    return {"accuracy": acc}


batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
print(f'logging steps: {logging_steps}')

model_name = f"finetuned-emotion"
training_args = TrainingArguments(output_dir=model_name,
                                  num_train_epochs=2,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False,
                                  logging_steps=logging_steps,
                                  log_level="error")

trainer = Trainer(model=model, args=training_args, 
                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"],
                  tokenizer=tokenizer)
trainer.train();

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")
