import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelClass(nn.Module):

    def __init__(self, model_name: str, num_labels: int, len_train_dl: int, lr: float, epochs: int) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.len_train_dl = len_train_dl
        self.lr = lr
        self.epochs = epochs
        self.model = BertForSequenceClassification.from_pretrained(self.model_name,num_labels=self.num_labels).to(DEVICE)
        # self.model = BertForSequenceClassification.from_pretrained(self.model_name,num_labels=1).to(DEVICE)
        # self.model = BertForSequenceClassification.from_pretrained(self.model_name).to(DEVICE)
        

    
    def _get_grouped_params(self, model):
        no_decay=["bias", "LayerNorm.weight"]
        params_with_wd, params_without_wd = [], []
        for n, p in self.model.named_parameters():
            if any(nd in n for nd in no_decay):
                params_without_wd.append(p)
            else:
                params_with_wd.append(p)
        return [{'params': params_with_wd, 'weight_decay': 0.1},
                {'params': params_without_wd, 'weight_decay': 0.0}]
    
    def opt_sch(self):
        optimizer = torch.optim.AdamW(self._get_grouped_params(self.model), lr=self.lr)
        total_steps = self.len_train_dl * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
        return optimizer, scheduler












# Create the learning rate scheduler.
