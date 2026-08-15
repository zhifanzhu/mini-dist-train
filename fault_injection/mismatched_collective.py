import os
from datetime import timedelta
import torch
import torch.distributed as dist


def init():
    backend = os.environ.get("MINI_DIST_BACKEND", "gloo")
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=8))
    return dist.get_rank(), dist.get_world_size()


rank, world = init()
x = torch.tensor([float(rank)])
if rank == 0:
    dist.all_reduce(x)
else:
    out = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(out, x)
dist.barrier()
