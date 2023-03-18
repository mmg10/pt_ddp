from timeit import default_timer as timer
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertConfig
# from transformers import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup

from model import ModelClass
from trainer import train
from dataset import datasets

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

TRAIN_DATA = pd.read_csv( config["TRAIN_DATA"])
VAL_DATA = pd.read_csv( config["VAL_DATA"])
TRAIN_DATA = TRAIN_DATA[:1024]
VAL_DATA = VAL_DATA[:512]

# TEST_DATA = pd.read_csv( config["TEST_DATA"])
MAX_LEN = config["MAX_LEN"]
MODEL = config["MODEL"]
NUM_LABELS = 2 # config["NUM_LABELS"]

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Dataset

# df = pd.read_csv("in_domain_train.tsv", delimiter='\t', header=None,names=['sentence_source', 'targets', 'label_notes', 'text'])
# df = df.loc[:, ['text', 'targets']]
# train_df, tnv_df = train_test_split(df, test_size=0.2)
# val_df, test_df = train_test_split(tnv_df, test_size=0.5)
# train_dl, val_dl = datasets(train_df, TRAIN_BATCH, val_df, VAL_BATCH, tokenizer, MAX_LEN)
# train_df.reset_index(inplace=True)
# val_df.reset_index(inplace=True)
# test_df.reset_index(inplace=True)

train_dl, val_dl = datasets(TRAIN_DATA, TRAIN_BATCH, VAL_DATA, VAL_BATCH, tokenizer, MAX_LEN)


# Set up model
model_class = ModelClass(model_name=MODEL, num_labels=NUM_LABELS, len_train_dl=len(train_dl), lr=LR, epochs=NUM_EPOCHS)

model = model_class.model
optimizer, scheduler = model_class.opt_sch()

# Start the timer

start_time = timer()


model_0_results = train(
    model=model,
    train_dataloader=train_dl,
    val_dataloader=val_dl,
    optimizer=optimizer,
    scheduler=scheduler,
    device=DEVICE,
    epochs=NUM_EPOCHS,
)

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")
