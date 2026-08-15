import pytest
import torch
import torch.nn as nn
from mini_dist.zero.common import partition_1d
from mini_dist.zero.stage3 import MiniZeRO3, ShardedTensor1D
from tests._dist_test_utils import run_gloo


def _worker(rank, world_size):
    for numel in (17, 2):  # non-divisible, then smaller than world_size
        full = torch.arange(numel, dtype=torch.float32)
        sharded = ShardedTensor1D.from_tensor(full)
        assert sharded.partition.original_numel == numel
        assert sharded.partition.padded_numel % world_size == 0
        assert sharded.local_shard.numel() == sharded.partition.shard_numel
        rebuilt = sharded.all_gather()
        assert rebuilt.shape == full.shape
        torch.testing.assert_close(rebuilt, full)


def _module_worker(rank, world_size):
    torch.manual_seed(9)
    module = nn.Linear(3, 2)
    reference = nn.Linear(3, 2)
    reference.load_state_dict(module.state_dict())
    model = MiniZeRO3(module)
    x = torch.tensor([[float(rank), 2.0, -1.0]])
    torch.testing.assert_close(model(x), reference(x))


def test_zero3_padding_and_exact_reconstruction():
    run_gloo(4, _worker)


def test_zero3_module_materializes_parameters_for_forward():
    run_gloo(2, _module_worker)


@pytest.mark.parametrize("numel,world_size", [(16, 4), (17, 4), (1, 4), (31, 8), (0, 4)])
def test_partition_metadata(numel, world_size):
    parts = [partition_1d(numel, r, world_size) for r in range(world_size)]
    padded = parts[0].padded_numel
    assert padded % world_size == 0
    assert all(p.original_numel == numel for p in parts)
    assert all(p.padded_numel == padded for p in parts)
    assert all(p.shard_numel == padded // world_size for p in parts)
    assert [(p.start, p.end) for p in parts] == [
        (r * (padded // world_size), (r + 1) * (padded // world_size)) for r in range(world_size)
    ]
