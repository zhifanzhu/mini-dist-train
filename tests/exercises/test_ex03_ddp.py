import torch
import torch.nn as nn
from mini_dist.ddp import MiniDDP
from tests._dist_test_utils import run_gloo


def _worker(rank, world_size):
    torch.manual_seed(0)
    model = MiniDDP(nn.Linear(3, 1, bias=False))
    x = torch.tensor([[1.0 + rank, 2.0, -1.0]])
    loss = model(x).sum()
    loss.backward()
    local_before = model.module.weight.grad.detach().clone()
    model.sync_gradients()

    # Reference mean local gradient can be derived analytically for linear(sum()).
    expected = torch.tensor([[sum(1.0 + r for r in range(world_size)) / world_size, 2.0, -1.0]])
    assert not torch.equal(local_before, expected) or world_size == 1
    torch.testing.assert_close(model.module.weight.grad, expected)

    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    opt.step()


def test_ddp_synchronizes_mean_gradient():
    run_gloo(2, _worker)
