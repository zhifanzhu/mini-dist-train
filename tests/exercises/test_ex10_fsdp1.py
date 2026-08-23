import copy

import torch
import torch.nn as nn
import torch.distributed as dist
from mini_dist.fsdp1 import MiniFSDP1
from mini_dist.zero.common import pad_flat, partition_1d
from tests._dist_test_utils import run_gloo


def _model():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))


def _flat_parameters(module):
    return torch.cat([parameter.detach().reshape(-1) for parameter in module.parameters()])


def _mean_reference_gradients(module, world_size):
    for parameter in module.parameters():
        dist.all_reduce(parameter.grad)
        parameter.grad.div_(world_size)


def _ownership_and_views_worker(rank, world_size):
    torch.manual_seed(42)
    base = _model()
    reference = copy.deepcopy(base)
    expected_names_and_shapes = {
        name: parameter.shape for name, parameter in reference.named_parameters()
    }
    expected_full = _flat_parameters(reference)
    partition = partition_1d(expected_full.numel(), rank, world_size)
    expected_padded = pad_flat(expected_full, partition.padded_numel)
    expected_local = expected_padded[partition.start : partition.end]

    model = MiniFSDP1(base)
    optimizer_parameters = list(model.parameters())

    assert len(optimizer_parameters) == 1, (
        "MiniFSDP1 must expose one permanent flat parameter to the optimizer; "
        f"found {len(optimizer_parameters)} registered parameters"
    )
    flat_parameter = optimizer_parameters[0]
    assert flat_parameter.numel() == partition.shard_numel, (
        "The optimizer-visible flat parameter must contain only the rank-local shard; "
        f"expected {partition.shard_numel} elements but found {flat_parameter.numel()}"
    )
    torch.testing.assert_close(
        flat_parameter.detach(),
        expected_local,
        msg="The optimizer-visible flat parameter has the wrong rank-local values",
    )
    assert not dict(base.named_parameters()), (
        "Original logical parameters must be removed from optimizer-visible module state "
        "while MiniFSDP1 is sharded"
    )

    x = torch.arange(4, dtype=torch.float32).reshape(1, 4) + rank
    output = model(x)
    expected_output = reference(x)
    torch.testing.assert_close(
        output,
        expected_output,
        msg="Forward must reconstruct the original parameter values from flat shards",
    )

    materialized = dict(base.named_parameters())
    assert set(materialized) == set(expected_names_and_shapes), (
        "Forward must install a view for every original logical parameter; "
        f"expected {set(expected_names_and_shapes)} but found {set(materialized)}"
    )
    for name, expected_shape in expected_names_and_shapes.items():
        assert materialized[name].shape == expected_shape, (
            f"Forward installed the wrong view shape for {name!r}: "
            f"expected {tuple(expected_shape)} but found {tuple(materialized[name].shape)}"
        )


def _backward_sharding_worker(rank, world_size):
    torch.manual_seed(42)
    base = _model()
    reference = copy.deepcopy(base)
    model = MiniFSDP1(base)
    flat_parameter = list(model.parameters())[0]
    x = torch.arange(4, dtype=torch.float32).reshape(1, 4) + rank

    reference(x).sum().backward()
    _mean_reference_gradients(reference, world_size)
    expected_full_gradient = torch.cat(
        [parameter.grad.detach().reshape(-1) for parameter in reference.parameters()]
    )
    partition = partition_1d(expected_full_gradient.numel(), rank, world_size)
    expected_padded_gradient = pad_flat(
        expected_full_gradient, partition.padded_numel
    )
    expected_local_gradient = expected_padded_gradient[
        partition.start : partition.end
    ]

    model(x).sum().backward()

    assert flat_parameter.grad is not None, (
        "Backward must write the reduced gradient shard to the optimizer-visible flat parameter"
    )
    assert flat_parameter.grad.numel() == partition.shard_numel, (
        "Backward must retain only the rank-local flat-gradient shard; "
        f"expected {partition.shard_numel} elements but found {flat_parameter.grad.numel()}"
    )
    torch.testing.assert_close(
        flat_parameter.grad,
        expected_local_gradient,
        msg="Backward produced the wrong mean-reduced flat-gradient shard",
    )
    assert flat_parameter.numel() == partition.shard_numel, (
        "After backward, the optimizer-visible flat parameter must remain rank-local"
    )
    assert not dict(base.named_parameters()), (
        "After backward, MiniFSDP1 must remove the gathered original-parameter views"
    )


def _two_step_worker(rank, world_size):
    torch.manual_seed(42)
    base = _model()
    reference = copy.deepcopy(base)
    model = MiniFSDP1(base)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)

    for step in range(2):
        x = torch.arange(4, dtype=torch.float32).reshape(1, 4) + rank + step
        expected_output = reference(x)
        output = model(x)
        torch.testing.assert_close(
            output,
            expected_output,
            msg=(
                f"Forward at step {step} must gather the flat shards updated by the "
                "previous optimizer step"
            ),
        )

        expected_output.sum().backward()
        _mean_reference_gradients(reference, world_size)
        reference_optimizer.step()
        reference_optimizer.zero_grad(set_to_none=True)

        output.sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        assert not dict(base.named_parameters()), (
            f"After backward at step {step}, original-parameter views must be removed"
        )

    flat_parameter = list(model.parameters())[0]
    gathered = torch.empty(
        flat_parameter.numel() * world_size,
        dtype=flat_parameter.dtype,
        device=flat_parameter.device,
    )
    dist.all_gather_into_tensor(gathered, flat_parameter.detach().contiguous())
    expected_full = _flat_parameters(reference)
    torch.testing.assert_close(
        gathered[: expected_full.numel()],
        expected_full,
        msg="Two optimizer steps must produce the same full parameters as data parallelism",
    )


def test_fsdp1_exposes_one_local_flat_parameter_and_materializes_views():
    run_gloo(2, _ownership_and_views_worker)


def test_fsdp1_backward_reduce_scatters_flat_gradient_and_reshards():
    run_gloo(2, _backward_sharding_worker)


def test_fsdp1_two_steps_use_optimizer_updated_flat_shards():
    run_gloo(2, _two_step_worker)
