from typing import Tuple, Dict, List
import pathlib
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


from transformers import BertTokenizer, BertConfig


class TextDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len: int = 512, eval_mode: bool = False):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.text = dataframe.text
        self.eval_mode = eval_mode 
        if self.eval_mode is False:
            self.targets = dataframe.targets
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        # text = str(self.text.iloc[index])
        # text = " ".join(text.split())
        text = str(self.text[index])

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )
        
        
        ids = inputs['input_ids'].flatten()
        mask = inputs['attention_mask'].flatten()
        token_type_ids = inputs["token_type_ids"].flatten()

        output = {
            'ids': ids,
            'mask': mask,
            'token_type_ids': token_type_ids,
        }

        if self.eval_mode is False:
            output['targets'] = torch.tensor(self.targets[index], dtype=torch.long)
                
        return output

        
def datasets(train_data, train_batchsize, val_data, val_batchsize, tokenizer, max_len, rank, world_size):


    train_dataset = TextDataset(train_data, tokenizer, max_len)
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True, rank=rank, num_replicas=world_size) 
 
    val_dataset = TextDataset(val_data, tokenizer, max_len)
    val_sampler = DistributedSampler(dataset=val_dataset, shuffle=False, rank=rank, num_replicas=world_size)

    train_dataloader = DataLoader(dataset=train_dataset,
                                        batch_size=train_batchsize,
                                        num_workers=4, 
                                        shuffle=True,
                                        pin_memory=True,
                                        sampler=train_sampler) 

    val_dataloader = DataLoader(dataset=val_dataset,
                                        batch_size=val_batchsize, 
                                        num_workers=4, 
                                        shuffle=False,
                                        pin_memory=True,
                                        sampler=val_sampler) 
    return train_dataloader, train_sampler, val_dataloader, val_sampler
