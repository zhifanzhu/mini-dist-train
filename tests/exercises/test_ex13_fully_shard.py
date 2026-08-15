import torch
import torch.nn as nn
from mini_dist.fsdp2 import fully_shard
from tests._dist_test_utils import run_gloo


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Linear(4, 4)
        self.block2 = nn.Linear(4, 4)
        self.head = nn.Linear(4, 1)

    def forward(self, x):
        return self.head(torch.relu(self.block2(torch.relu(self.block1(x)))))


def _worker(rank, world_size):
    torch.manual_seed(0)
    model = TinyModel()
    fully_shard(model.block1)
    fully_shard(model.block2)
    fully_shard(model)

    # Educational API stores owned parameter names for introspection.
    groups = [model.block1._mini_fsdp_group, model.block2._mini_fsdp_group, model._mini_fsdp_group]
    owned = [set(g.owned_parameter_ids) for g in groups]
    assert owned[0].isdisjoint(owned[1])
    assert owned[0].isdisjoint(owned[2])
    assert owned[1].isdisjoint(owned[2])
    # Root should own the head parameters, but not child-owned block parameters.
    assert len(owned[2]) == 2

    opt = torch.optim.SGD(model.parameters(), lr=0.01)  # optimizer after fully_shard
    x = torch.ones(2, 4) * (rank + 1)
    loss = model(x).sum()
    loss.backward()
    opt.step()


def test_bottom_up_fully_shard_ownership_and_training():
    run_gloo(2, _worker)
