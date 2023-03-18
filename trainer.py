import pdb
import time

from rich.console import Console
from rich.table import Table
from rich import print as rprint
# from rich.progress import track
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    MofNCompleteColumn,
    TimeRemainingColumn
)

import torch
import torch.nn as nn

def train_loop(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               optimizer: torch.optim.Optimizer,
               scheduler,
               device,
               pb,
               task_train):
    
    model.train()
    train_loss, train_acc = 0, 0

    # for (x, y) in track(dataloader, "Training"):
    for ind, batch in enumerate(dataloader):
        model.zero_grad(set_to_none=True)
        # or optimizer.zero_grad()
        input_ids = batch['ids'].to(device)
        input_mask =  batch['mask'].to(device)
        token_type_ids =  batch['token_type_ids'].to(device)
        labels = batch['targets'].to(device)
        # pdb.set_trace()
        outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask, labels=labels)
        loss = outputs['loss']

        y_pred = torch.argmax(outputs['logits'], dim=1)

        train_loss += loss.item() 
        train_acc += (y_pred == labels).sum().item()/len(y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        pb.update(task_id=task_train, completed=ind+1)
        
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader) *100
    return train_loss, train_acc


def val_loop(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              device,
              pb,
              task_eval):

    model.eval() 
    val_loss, val_acc = 0, 0
    
    
    # for (x, y) in track(dataloader, "Validating"):
    for ind, batch in enumerate(dataloader):
        input_ids = batch['ids'].to(device)
        input_mask =  batch['mask'].to(device)
        token_type_ids =  batch['token_type_ids'].to(device)
        labels = batch['targets'].to(device)
        
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=input_mask, labels=labels)
        
        loss = outputs['loss']

        y_pred = torch.argmax(outputs['logits'], dim=1)

        val_loss += loss.item()
        val_acc += ((y_pred == labels).sum().item()/len(y_pred))
        pb.update(task_id=task_eval, completed=ind+1)
            
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader) *100
    return val_loss, val_acc

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          scheduler,
          device,
          epochs: int = 5):
    
    console = Console()
    table = Table(title='Model Training/Validation Logs')
    table.add_column('epoch', style='blue')
    table.add_column('Train Acc', style='green', justify='center')
    table.add_column('Train Loss', style='red', justify='center')
    table.add_column('Valid Acc', style='green', justify='center')
    table.add_column('Valid Loss', style='blue', justify='center')
    table.add_column('Time', style='red', justify='right')
    

    results =  {"train_loss": [],
                "train_acc": [],
                "val_loss": [],
                "val_acc": []
            }
    
    pb = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn()
    )
    with pb:
        task_total = pb.add_task('total', total=epochs)
        task_train = pb.add_task('train', total=len(train_dataloader))
        task_eval = pb.add_task('eval', total=len(val_dataloader))
        

        for epoch in range(epochs):
            start_time = time.time()

            train_loss, train_acc = train_loop(model=model,
                                            dataloader=train_dataloader,
                                            optimizer=optimizer,
                                            scheduler=scheduler,
                                            device=device,
                                            pb=pb,
                                            task_train=task_train)
            pb.reset(task_train)
            val_loss, val_acc = val_loop(model=model,
                                            dataloader=val_dataloader,
                                            device=device,
                                            pb=pb,
                                            task_eval=task_eval)
            pb.reset(task_eval)

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            time_int = f'{epoch_mins}m {epoch_secs}s'
            table.add_row(str(epoch+1), str(round(train_acc,2)), str(round(train_loss,2)), str(round(val_acc,2)), str(round(val_loss,2)), time_int)
            

            rprint(f"[bold red]Epoch:[/bold red] {epoch+1}, [bold red]Train Acc:[/bold red] {str(round(train_acc,2))}, [bold red]Train Loss:[/bold red] {str(round(train_loss,2))}, [bold red]Val Acc:[/bold red] {str(round(val_acc,2))}, [bold red]Val Loss:[/bold red] {str(round(val_loss,2))}, [bold red]Time:[/bold red] {time_int}")


            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["val_loss"].append(val_loss)
            results["val_acc"].append(val_acc)
            pb.update(task_id=task_total, completed=epoch+1)

    console.print(table)
    return results