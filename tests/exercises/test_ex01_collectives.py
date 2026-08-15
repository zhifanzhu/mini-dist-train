import torch
from mini_dist.collectives import all_reduce_mean, all_gather_fixed, reduce_scatter_sum
from tests._dist_test_utils import run_gloo


def _worker(rank, world_size):
    x = torch.tensor([float(rank + 1), float(2 * (rank + 1))])
    out = all_reduce_mean(x.clone())
    expected = torch.tensor([(world_size + 1) / 2, world_size + 1.0])
    torch.testing.assert_close(out, expected)

    local = torch.tensor([rank, rank + 10], dtype=torch.float32)
    gathered = all_gather_fixed(local)
    expected_g = torch.cat([torch.tensor([r, r + 10], dtype=torch.float32) for r in range(world_size)])
    torch.testing.assert_close(gathered, expected_g)

    flat = torch.arange(world_size * 4, dtype=torch.float32) + rank
    shard = reduce_scatter_sum(flat)
    total = sum(torch.arange(world_size * 4, dtype=torch.float32) + r for r in range(world_size))
    expected_s = total.chunk(world_size)[rank]
    torch.testing.assert_close(shard, expected_s)


def test_collective_contracts():
    run_gloo(2, _worker)
