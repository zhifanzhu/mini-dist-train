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
    raise RuntimeError("simulated exception/OOM before next collective")
x = torch.tensor([float(rank)])
dist.reduce_scatter_tensor(torch.empty(1), torch.arange(world, dtype=torch.float32))
