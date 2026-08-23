""" Usage

> torchrun \
          --nnodes=1 \
          --nproc-per-node=2 \
          --master-addr=127.0.0.1 \
          --master-port=29500 \
          mydriver.py
"""

import os
import torch
import torch.distributed as dist
from mini_dist.collectives import ring_all_reduce_sum, tree_all_reduce_sum

dist.init_process_group(backend="gloo")

rank = dist.get_rank()
world_size = dist.get_world_size()

print(
    f"rank={rank}, world_size={world_size}, "
    f"pid={os.getpid()}"
)

# x = torch.tensor([0])
# x = torch.arange(8)
x = torch.arange(1, world_size+1) * (10**rank)
print(f"{rank=}, {x=}")

# A: a0, b0, c0, d0 = [1, 2, 3, 4]
# B: a1, b1, c1, d1 = [10, 20, 30, 40]
# exp final: A=B= [11, 22, 33, 44]

# ring_all_reduce_sum(x)

# print(f"{rank=}, {x=}")

# tree_all_reduce_sum(x)

# print(f"{rank=}, {x=}")

a = torch.zeros([6]).float()
tensor_list = list(a.chunk(2))
tensor = (torch.arange(3) + 10 * rank).float()
print(f"{rank=} {tensor=}")
dist.all_gather(tensor_list, tensor)
# expect that a gets updated
print(a)


from torch.nn.parallel import DistributedDataParallel as DDP
ddp_model = DDP(torch.nn.Linear(3, 4))

""" Deadlock version """
# if rank == 0:
#     dist.send(x, dst=1)
#     dist.recv(x, src=1)
# if rank == 1:
#     dist.send(x, dst=0)
#     dist.recv(x, src=0)

""" OK ? """
rs = dist.isend(x, dst=(rank + 1) % world_size)
rr = dist.irecv(x, src=(rank + world_size - 1) % world_size)
rr.wait()
rs.wait()

dist.destroy_process_group()
