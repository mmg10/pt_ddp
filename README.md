# PyTorch DDP


Repo for DDP in PyTorch


Code:

```bash
export OMP_NUM_THREADS=16
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=101 --rdzv_endpoint="localhost:5601" main_pt.py
```


# to do

```
test on a multi-gpu system
tensorboard --logdir=./fsdp --bind_all --host=0.0.0.0 --port 8000
```

```
pip install nvidia-pyindex
pip install nvidia-dlprof
```