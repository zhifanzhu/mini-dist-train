import copy
import torch
import torch.nn as nn
from mini_dist.zero.stage1 import Zero1Optimizer
from tests._dist_test_utils import run_gloo


def _worker(rank, world_size):
    torch.manual_seed(123)
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))
    reference = copy.deepcopy(model)
    total_numel = sum(p.numel() for p in model.parameters())
    for p in model.parameters():
        p.grad = torch.ones_like(p) * 0.5  # already-reduced replicated grad
    for p in reference.parameters():
        p.grad = torch.ones_like(p) * 0.5
    opt = Zero1Optimizer(model.parameters(), torch.optim.Adam, lr=1e-2)
    reference_opt = torch.optim.Adam(reference.parameters(), lr=1e-2)

    assert opt.partition.original_numel == total_numel
    assert opt.local_master_param.numel() == opt.partition.shard_numel
    optimizer_owned = sum(p.numel() for g in opt.optimizer.param_groups for p in g["params"])
    assert optimizer_owned == opt.partition.shard_numel
    assert optimizer_owned < opt.partition.padded_numel or world_size == 1

    reference_opt.step()
    opt.step()
    assert len(opt.optimizer.state) == 1  # Adam state exists only for local master partition

    # The replicated result must equal an ordinary optimizer update, not merely
    # agree across ranks.
    for p, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, expected)
        gathered = [torch.empty_like(p) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, p)
        for other in gathered[1:]:
            torch.testing.assert_close(other, gathered[0])


def test_zero1_shards_optimizer_storage_but_replicates_updated_parameters():
    run_gloo(2, _worker)
