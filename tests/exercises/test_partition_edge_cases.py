import pytest
from mini_dist.zero.common import partition_1d


@pytest.mark.parametrize("numel,world_size", [(16,4), (17,4), (1,4), (31,8), (0,4)])
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
