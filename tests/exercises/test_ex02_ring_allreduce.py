import torch
import torch.distributed as dist
from mini_dist.collectives import ring_all_reduce_sum
from tests._dist_test_utils import run_gloo


def _worker(rank, world_size):
    x = torch.arange(world_size * 3, dtype=torch.float32) + 10 * rank
    out = ring_all_reduce_sum(x)
    expected = sum(torch.arange(world_size * 3, dtype=torch.float32) + 10 * r for r in range(world_size))
    torch.testing.assert_close(out, expected)


def _subgroup_worker(rank, world_size):
    members = [1, 3]
    group = dist.new_group(ranks=members)

    if rank in members:
        x = torch.arange(6, dtype=torch.float32) + 10 * rank
        out = ring_all_reduce_sum(x, group=group)
        expected = sum(torch.arange(6, dtype=torch.float32) + 10 * r for r in members)
        torch.testing.assert_close(out, expected)

    dist.barrier()


def test_ring_matches_allreduce_sum():
    run_gloo(4, _worker)


def test_ring_handles_the_single_rank_identity_case():
    run_gloo(1, _worker)


def test_ring_uses_group_relative_peers():
    run_gloo(4, _subgroup_worker)
