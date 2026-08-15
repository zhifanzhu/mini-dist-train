import torch
import torch.nn as nn
from mini_dist.zero.stage1 import Zero1Optimizer
from tests._dist_test_utils import run_gloo


def _worker(rank, world_size):
    torch.manual_seed(123)
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))
    total_numel = sum(p.numel() for p in model.parameters())
    for p in model.parameters():
        p.grad = torch.ones_like(p) * 0.5  # already-reduced replicated grad
    opt = Zero1Optimizer(model.parameters(), torch.optim.Adam, lr=1e-2)

    assert opt.partition.original_numel == total_numel
    assert opt.local_master_param.numel() == opt.partition.shard_numel
    optimizer_owned = sum(p.numel() for g in opt.optimizer.param_groups for p in g["params"])
    assert optimizer_owned == opt.partition.shard_numel
    assert optimizer_owned < opt.partition.padded_numel or world_size == 1

    opt.step()
    assert len(opt.optimizer.state) == 1  # Adam state exists only for local master partition

    # Every rank must end with identical replicated parameters.
    for p in model.parameters():
        gathered = [torch.empty_like(p) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, p)
        for other in gathered[1:]:
            torch.testing.assert_close(other, gathered[0])


def test_zero1_shards_optimizer_storage_but_replicates_updated_parameters():
    run_gloo(2, _worker)
