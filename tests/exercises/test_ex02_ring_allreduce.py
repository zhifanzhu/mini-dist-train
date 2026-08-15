import torch
import torch.distributed as dist
import mini_dist.collectives as collectives
from mini_dist.collectives import ring_all_reduce_sum
from tests._dist_test_utils import run_gloo


def _worker(rank, world_size):
    x = torch.arange(world_size * 3, dtype=torch.float32) + 10 * rank
    original_storage = x.data_ptr()
    sent_numels = []
    actual_send = collectives.dist.send

    def tracked_send(tensor, *args, **kwargs):
        sent_numels.append(tensor.numel())
        return actual_send(tensor, *args, **kwargs)

    collectives.dist.send = tracked_send
    try:
        out = ring_all_reduce_sum(x)
    finally:
        collectives.dist.send = actual_send

    expected = sum(torch.arange(world_size * 3, dtype=torch.float32) + 10 * r for r in range(world_size))
    assert out is x
    assert out.data_ptr() == original_storage
    torch.testing.assert_close(out, expected)

    if world_size > 1:
        assert sent_numels == [x.numel() // world_size] * (2 * (world_size - 1))


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
