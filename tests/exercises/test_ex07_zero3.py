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


def _backward_worker(rank, world_size):
    torch.manual_seed(17)
    module = nn.Linear(3, 1)
    reference = nn.Linear(3, 1)
    reference.load_state_dict(module.state_dict())
    model = MiniZeRO3(module)

    parameters = list(module.named_parameters())
    reference_parameters = list(reference.named_parameters())
    locally_sharded = all(
        parameter.numel()
        == partition_1d(expected.numel(), rank, world_size).shard_numel
        for (_, parameter), (_, expected) in zip(parameters, reference_parameters)
    )
    sharded_on_all_ranks = torch.tensor(int(locally_sharded))
    torch.distributed.all_reduce(sharded_on_all_ranks, op=torch.distributed.ReduceOp.MIN)
    assert sharded_on_all_ranks.item(), (
        "MiniZeRO3 must keep only rank-local parameter storage outside computation; "
        "retaining full parameters makes stage 3 equivalent to stage 2"
    )

    x = torch.tensor([[float(rank + 1), 2.0, -1.0]])
    reference(x).sum().backward()
    for parameter in reference.parameters():
        torch.distributed.all_reduce(parameter.grad)
        parameter.grad.div_(world_size)

    model(x).sum().backward()

    parameters = list(module.named_parameters())
    locally_resharded = all(
        parameter.numel()
        == partition_1d(expected.numel(), rank, world_size).shard_numel
        and parameter.grad is not None
        and parameter.grad.numel()
        == partition_1d(expected.numel(), rank, world_size).shard_numel
        for (_, parameter), (_, expected) in zip(parameters, reference_parameters)
    )
    resharded_on_all_ranks = torch.tensor(int(locally_resharded))
    torch.distributed.all_reduce(resharded_on_all_ranks, op=torch.distributed.ReduceOp.MIN)
    assert resharded_on_all_ranks.item(), (
        "After backward, MiniZeRO3 must restore rank-local parameter storage "
        "and leave only the rank-local reduced gradient shard"
    )

    for (name, parameter), (_, expected) in zip(parameters, reference_parameters):
        partition = partition_1d(expected.numel(), rank, world_size)
        padded_parameter = torch.zeros(
            partition.padded_numel, dtype=expected.dtype, device=expected.device
        )
        padded_parameter[: expected.numel()] = expected.detach().reshape(-1)
        expected_parameter_shard = padded_parameter[partition.start : partition.end]
        torch.testing.assert_close(
            parameter.detach().reshape(-1),
            expected_parameter_shard,
            msg=f"Backward must restore the authoritative parameter shard for {name!r}",
        )

        padded_gradient = torch.zeros_like(padded_parameter)
        padded_gradient[: expected.grad.numel()] = expected.grad.reshape(-1)
        expected_gradient_shard = padded_gradient[partition.start : partition.end]
        torch.testing.assert_close(
            parameter.grad.reshape(-1),
            expected_gradient_shard,
            msg=f"Backward must reduce-scatter the mean gradient for {name!r}",
        )


def test_zero3_padding_and_exact_reconstruction():
    run_gloo(4, _worker)


def test_zero3_module_materializes_parameters_for_forward():
    run_gloo(2, _module_worker)


def test_zero3_backward_reduce_scatters_gradients_and_reshards_parameters():
    run_gloo(2, _backward_worker)


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
