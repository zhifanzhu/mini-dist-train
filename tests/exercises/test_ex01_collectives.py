import pytest
import torch
import torch.distributed as dist
from mini_dist.collectives import all_reduce_mean, all_gather_fixed, reduce_scatter_sum
from tests._dist_test_utils import run_gloo


def _worker(rank, world_size):
    x = torch.tensor([float(rank + 1), float(2 * (rank + 1))])
    original_storage = x.data_ptr()
    out = all_reduce_mean(x)
    expected = torch.tensor([(world_size + 1) / 2, world_size + 1.0])
    assert out is x
    assert out.data_ptr() == original_storage
    torch.testing.assert_close(out, expected)

    local = torch.tensor([[rank, rank + 10], [rank + 20, rank + 30]], dtype=torch.float32)
    gathered = all_gather_fixed(local)
    expected_g = torch.cat(
        [torch.tensor([[r, r + 10], [r + 20, r + 30]], dtype=torch.float32) for r in range(world_size)]
    )
    torch.testing.assert_close(gathered, expected_g)

    flat = torch.arange(world_size * 4, dtype=torch.float32) + rank
    shard = reduce_scatter_sum(flat)
    total = sum(torch.arange(world_size * 4, dtype=torch.float32) + r for r in range(world_size))
    expected_s = total.chunk(world_size)[rank]
    torch.testing.assert_close(shard, expected_s)


def _subgroup_worker(rank, world_size):
    members = [1, 3]
    group = dist.new_group(ranks=members)

    if rank in members:
        group_rank = dist.get_rank(group)
        group_world_size = dist.get_world_size(group)

        x = torch.tensor([float(rank + 1)])
        out = all_reduce_mean(x, group=group)
        torch.testing.assert_close(out, torch.tensor([3.0]))

        local = torch.tensor([[rank, rank + 10]], dtype=torch.float32)
        gathered = all_gather_fixed(local, group=group)
        expected_g = torch.tensor([[1, 11], [3, 13]], dtype=torch.float32)
        torch.testing.assert_close(gathered, expected_g)

        flat = torch.arange(group_world_size * 3, dtype=torch.float32) + rank
        shard = reduce_scatter_sum(flat, group=group)
        total = sum(torch.arange(group_world_size * 3, dtype=torch.float32) + r for r in members)
        torch.testing.assert_close(shard, total.chunk(group_world_size)[group_rank])

    dist.barrier()


def _invalid_reduce_scatter_worker(rank, world_size):
    with pytest.raises(ValueError):
        reduce_scatter_sum(torch.arange(world_size * 3 + 1, dtype=torch.float32))

    with pytest.raises(ValueError):
        reduce_scatter_sum(torch.zeros(world_size, 2))


def test_collective_contracts():
    run_gloo(3, _worker)


def test_collectives_use_the_supplied_process_group():
    run_gloo(4, _subgroup_worker)


def test_reduce_scatter_rejects_non_fixed_size_inputs_before_communication():
    run_gloo(2, _invalid_reduce_scatter_worker)
