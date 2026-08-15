import torch
from mini_dist.collectives import ring_all_reduce_sum
from tests._dist_test_utils import run_gloo


def _worker(rank, world_size):
    x = torch.arange(8, dtype=torch.float32) + 10 * rank
    out = ring_all_reduce_sum(x.clone())
    expected = sum(torch.arange(8, dtype=torch.float32) + 10 * r for r in range(world_size))
    torch.testing.assert_close(out, expected)


def test_ring_matches_allreduce_sum():
    run_gloo(2, _worker)
