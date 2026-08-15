import torch
import torch.distributed as dist
from mini_dist.collectives import tree_all_reduce_sum
from tests._dist_test_utils import run_gloo


def _worker(rank, world_size):
    # Five ranks exercises a non-power-of-two tree without imposing a layout.
    x = torch.arange(7, dtype=torch.float32) + 10 * rank
    out = tree_all_reduce_sum(x)
    expected = sum(torch.arange(7, dtype=torch.float32) + 10 * r for r in range(world_size))
    torch.testing.assert_close(out, expected)


def _subgroup_worker(rank, world_size):
    members = [1, 3, 4]
    group = dist.new_group(ranks=members)

    if rank in members:
        x = torch.arange(5, dtype=torch.float32) + 10 * rank
        out = tree_all_reduce_sum(x, group=group)
        expected = sum(torch.arange(5, dtype=torch.float32) + 10 * r for r in members)
        torch.testing.assert_close(out, expected)

    dist.barrier()


def test_tree_matches_allreduce_sum_for_non_power_of_two_world_size():
    run_gloo(5, _worker)


def test_tree_handles_the_single_rank_identity_case():
    run_gloo(1, _worker)


def test_tree_uses_group_relative_peers():
    run_gloo(5, _subgroup_worker)
