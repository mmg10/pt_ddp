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
import torch.distributed as dist



import torch
from abc import abstractmethod
from numpy import inf


class TrainerClass:
    """
    Base class for all trainers
    """
    def __init__(self, model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader,
          train_sampler, 
          val_dataloader: torch.utils.data.DataLoader, 
          val_sampler,
          optimizer: torch.optim.Optimizer,
          scheduler,
          device,
          epochs: int = 5,
          rank: int = 1,
          local_rank: int = 0,
          run_id = 0):
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.train_sampler = train_sampler
        self.val_dataloader = val_dataloader
        self.val_sampler = val_sampler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_epochs = epochs
        self.rank = rank
        self.local_rank = local_rank
        self.run_id = run_id      

        

        if self.rank ==0:
            self.pb = Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    MofNCompleteColumn(),
                    TimeRemainingColumn()
                )
            self.task_total = self.pb.add_task('total', total=self.max_epochs)
            self.task_train = self.pb.add_task('train', total=len(self.train_dataloader))
            self.task_eval = self.pb.add_task('eval', total=len(self.val_dataloader))

    @abstractmethod
    def _train_epoch(self):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """

        self.model.train()
        # train_loss, train_acc = 0, 0cccc
        train_acc = 0
        ddp_train_loss = torch.zeros(2).to(self.local_rank)

        # for (x, y) in track(dataloader, "Training"):
        for ind, batch in enumerate(self.train_dataloader):
            self.model.zero_grad(set_to_none=True)
            # or optimizer.zero_grad()
            input_ids = batch['ids'].to(self.device)
            input_mask =  batch['mask'].to(self.device)
            token_type_ids =  batch['token_type_ids'].to(self.device)
            labels = batch['targets'].to(self.device)
            outputs = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask, labels=labels)
            loss = outputs['loss']

            y_pred = torch.argmax(outputs['logits'], dim=1)

            ddp_train_loss[0] += loss.item() 
            ddp_train_loss[1] += len(batch)
            train_acc += (y_pred == labels).sum().item()/len(y_pred)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            if self.rank == 0:
                self.pb.update(task_id=self.task_train, completed=ind+1)
        
        # rprint(f'rank: {self.rank}, loss: {ddp_train_loss}')
        dist.all_reduce(ddp_train_loss, op=dist.ReduceOp.SUM)
        # rprint(f'rank: {self.rank}, loss: {ddp_train_loss}')
        if self.rank==1:
            print(f'rank: {self.rank}, loss: {ddp_train_loss}')
        ddp_train_loss = (ddp_train_loss[0] / ddp_train_loss[1]).item()
        train_acc = train_acc / len(self.train_dataloader) *100
        return ddp_train_loss, train_acc
        
    @abstractmethod
    def _val_epoch(self):
        """
        Validation logic for an epoch

        :param epoch: Current epoch number
        """

        self.model.eval() 
        val_acc = 0
        ddp_val_loss = torch.zeros(2).to(self.local_rank)

        for ind, batch in enumerate(self.val_dataloader):
            input_ids = batch['ids'].to(self.device)
            input_mask =  batch['mask'].to(self.device)
            token_type_ids =  batch['token_type_ids'].to(self.device)
            labels = batch['targets'].to(self.device)
            
            with torch.inference_mode():
                outputs = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask, labels=labels)

            loss = outputs['loss']

            y_pred = torch.argmax(outputs['logits'], dim=1)

            ddp_val_loss[0] += loss.item() 
            ddp_val_loss[1] += len(batch)
            val_acc += (y_pred == labels).sum().item()/len(y_pred)
            if self.rank == 0:
                self.pb.update(task_id=self.task_eval, completed=ind+1)
        
        # print(f'rank: {self.rank}, loss: {ddp_val_loss}')
        dist.all_reduce(ddp_val_loss, op=dist.ReduceOp.SUM)
        # print(f'rank: {self.rank}, loss: {ddp_val_loss}')
        ddp_val_loss = (ddp_val_loss[0] / ddp_val_loss[1]).item()
        val_acc = val_acc / len(self.val_dataloader) *100
        return ddp_val_loss, val_acc

    def _epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def fit(self):
    
        if self.rank == 0:
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

            best_train_accuracy = float("-inf")
            best_val_loss = float("inf")

        
        cm = self.pb if self.rank == 0 else contextlib.nullcontext()
        with cm:            
            print(f'rank: {rank}, cm: {cm}')
            for epoch in range(1,self.max_epochs+1):
                start_time = time.time()
                self.train_sampler.set_epoch(epoch)
                
                train_loss, train_acc = self._train_epoch()
                if self.rank == 0:
                    self.pb.reset(self.task_train)
                # wait_for_everyone
                torch.distributed.barrier()

                # if rank == 0:
                self.val_sampler.set_epoch(epoch)
                val_loss, val_acc = self._val_epoch()

                if self.rank == 0:
                    self.pb.reset(self.task_eval)

                # wait_for_everyone
                torch.distributed.barrier()

                end_time = time.time()
                epoch_mins, epoch_secs = self._epoch_time(start_time, end_time)
                time_int = f'{epoch_mins}m {epoch_secs}s'
                table.add_row(str(epoch), str(round(train_acc,2)), str(round(train_loss,2)), str(round(val_acc,2)), str(round(val_loss,2)), time_int)
                
                results["train_loss"].append(train_loss)
                results["train_acc"].append(train_acc)
                results["val_loss"].append(val_loss)
                results["val_acc"].append(val_acc)

                rprint(f"[bold red]Epoch:[/bold red] {epoch}, [bold red]Train Acc:[/bold red] {str(round(train_acc,2))}, [bold red]Train Loss:[/bold red] {str(round(train_loss,2))}, [bold red]Val Acc:[/bold red] {str(round(val_acc,2))}, [bold red]Val Loss:[/bold red] {str(round(val_loss,2))}, [bold red]Time:[/bold red] {time_int}")

                if self.rank != 0:
                    print(f'======= rank is {self.rank} here')
                if self.rank == 0:
                    self.pb.update(task_id=self.task_total, completed=epoch)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        print(f"-->>>> New Val Loss Record: {best_val_loss}")
                        save_name = f"{self.run_id}-epoch-{epoch}.pt"
                        torch.save(self.model.state_dict(), save_name) # or should it be model.module.state_dict() ??               
        if self.rank == 0:
            console.print(table)
        return results



