import torch
from mini_dist.zero.stage3 import ShardedTensor1D
from tests._dist_test_utils import run_gloo


def _worker(rank, world_size):
    # Deliberately non-divisible: 17 elements over 4 ranks in the main edge case.
    full = torch.arange(17, dtype=torch.float32)
    sharded = ShardedTensor1D.from_tensor(full)
    assert sharded.partition.original_numel == 17
    assert sharded.partition.padded_numel % world_size == 0
    assert sharded.local_shard.numel() == sharded.partition.shard_numel
    rebuilt = sharded.all_gather()
    assert rebuilt.shape == full.shape
    torch.testing.assert_close(rebuilt, full)


def test_zero3_padding_and_exact_reconstruction():
    run_gloo(4, _worker)
