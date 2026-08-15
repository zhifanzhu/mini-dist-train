import torch
import torch.nn as nn
from mini_dist.fsdp1 import MiniFSDP1
from tests._dist_test_utils import run_gloo


def _worker(rank, world_size):
    torch.manual_seed(42)
    base = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    model = MiniFSDP1(base)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.arange(4, dtype=torch.float32).reshape(1,4) + rank
    loss = model(x).sum()
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


def test_fsdp1_normal_pytorch_training_loop_runs():
    run_gloo(2, _worker)
