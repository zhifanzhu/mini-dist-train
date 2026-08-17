import copy

import pytest
import torch
import torch.nn as nn

from mini_dist.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    copy_to_tensor_parallel_region,
    gather_from_tensor_parallel_region,
    reduce_from_tensor_parallel_region,
    scatter_to_tensor_parallel_region,
)
from tests._dist_test_utils import run_gloo


def _collective_worker(rank, world_size):
    replicated = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    replicated.requires_grad_(True)
    copied = copy_to_tensor_parallel_region(replicated, group=None)
    torch.testing.assert_close(copied, replicated)
    (copied * (rank + 1)).sum().backward()
    expected_scale = world_size * (world_size + 1) / 2
    torch.testing.assert_close(
        replicated.grad, torch.full_like(replicated, expected_scale)
    )

    partial = torch.full((2, 3), float(rank + 1), requires_grad=True)
    reduced = reduce_from_tensor_parallel_region(partial, group=None)
    torch.testing.assert_close(
        reduced, torch.full_like(reduced, expected_scale)
    )
    reduced.sum().backward()
    torch.testing.assert_close(partial.grad, torch.ones_like(partial))

    full = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    full.requires_grad_(True)
    local = scatter_to_tensor_parallel_region(full, group=None)
    torch.testing.assert_close(local, full.detach().chunk(world_size, dim=-1)[rank])
    gathered = gather_from_tensor_parallel_region(local, group=None)
    torch.testing.assert_close(gathered, full)
    gathered.square().sum().backward()
    torch.testing.assert_close(full.grad, 2 * full.detach())


def test_autograd_layout_transitions_have_the_correct_duals():
    run_gloo(2, _collective_worker)


def _linear_worker(rank, world_size):
    torch.manual_seed(17)

    column_source = nn.Linear(6, 8, bias=True)
    column_reference = copy.deepcopy(column_source)
    column = ColumnParallelLinear(column_source, gather_output=False)
    assert column.weight.shape == (8 // world_size, 6)
    assert column.bias.shape == (8 // world_size,)
    assert sum(p.numel() for p in column.parameters()) == (8 // world_size) * 7

    x_reference = torch.linspace(-1, 1, 24).reshape(2, 2, 6).requires_grad_(True)
    x_parallel = x_reference.detach().clone().requires_grad_(True)
    expected_full = column_reference(x_reference)
    got_local = column(x_parallel)
    torch.testing.assert_close(
        got_local, expected_full.detach().chunk(world_size, dim=-1)[rank]
    )

    expected_full.square().sum().backward()
    got_local.square().sum().backward()
    torch.testing.assert_close(x_parallel.grad, x_reference.grad)
    torch.testing.assert_close(
        column.weight.grad,
        column_reference.weight.grad.chunk(world_size, dim=0)[rank],
    )
    torch.testing.assert_close(
        column.bias.grad,
        column_reference.bias.grad.chunk(world_size, dim=0)[rank],
    )

    gathered_column = ColumnParallelLinear(
        copy.deepcopy(column_source), gather_output=True
    )
    torch.testing.assert_close(
        gathered_column(x_parallel.detach()), column_source(x_parallel.detach())
    )

    row_source = nn.Linear(8, 5, bias=True)
    row_reference = copy.deepcopy(row_source)
    row = RowParallelLinear(row_source)
    assert row.weight.shape == (5, 8 // world_size)
    assert row.bias.shape == (5,)
    assert sum(p.numel() for p in row.parameters()) == 5 * (8 // world_size) + 5

    row_x_reference = torch.linspace(-0.7, 0.9, 32).reshape(2, 2, 8)
    row_x_reference.requires_grad_(True)
    row_x_local = (
        row_x_reference.detach()
        .chunk(world_size, dim=-1)[rank]
        .clone()
        .requires_grad_(True)
    )
    row_expected = row_reference(row_x_reference)
    row_got = row(row_x_local)
    torch.testing.assert_close(row_got, row_expected)

    row_expected.square().sum().backward()
    row_got.square().sum().backward()
    torch.testing.assert_close(
        row_x_local.grad, row_x_reference.grad.chunk(world_size, dim=-1)[rank]
    )
    torch.testing.assert_close(
        row.weight.grad,
        row_reference.weight.grad.chunk(world_size, dim=1)[rank],
    )
    torch.testing.assert_close(row.bias.grad, row_reference.bias.grad)

    with pytest.raises(ValueError, match="out_features"):
        ColumnParallelLinear(nn.Linear(4, 7))
    with pytest.raises(ValueError, match="in_features"):
        RowParallelLinear(nn.Linear(7, 4))


def test_column_and_row_parallel_linears_match_unsharded_autograd():
    run_gloo(2, _linear_worker)
