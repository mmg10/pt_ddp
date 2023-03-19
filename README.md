# PyTorch DDP


Repo for DDP in PyTorch


Code:

```bash
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=101 --rdzv_endpoint="localhost:5601" main_pt.py
```


# to do

<!-- make loss have loss and batch size -->
<!-- dist all reduce -->