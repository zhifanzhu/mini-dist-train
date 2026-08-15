import os
from datetime import timedelta
import torch
import torch.distributed as dist


def init():
    backend = os.environ.get("MINI_DIST_BACKEND", "gloo")
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=8))
    return dist.get_rank(), dist.get_world_size()


rank, world = init()
if rank == 1:
    raise SystemExit("intentional rank exit")
x = torch.tensor([float(rank)])
dist.all_reduce(x)
