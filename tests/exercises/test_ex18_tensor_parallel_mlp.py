import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_dist.tensor_parallel import TensorParallelMLP
from tests._dist_test_utils import run_gloo


def _tp_shard(name, tensor, rank, world_size):
    if name in {"up_proj.weight", "up_proj.bias"}:
        return tensor.chunk(world_size, dim=0)[rank]
    if name == "down_proj.weight":
        return tensor.chunk(world_size, dim=1)[rank]
    if name == "down_proj.bias":
        return tensor
    raise AssertionError(f"unexpected parameter {name}")


def _worker(rank, world_size):
    torch.manual_seed(18)
    up_source = nn.Linear(6, 10, bias=True)
    down_source = nn.Linear(10, 4, bias=True)
    up_reference = copy.deepcopy(up_source)
    down_reference = copy.deepcopy(down_source)

    model = TensorParallelMLP(up_source, down_source, activation=F.gelu)
    assert model.up_proj.weight.shape == (10 // world_size, 6)
    assert model.down_proj.weight.shape == (4, 10 // world_size)

    local_hidden_sizes = []
    handle = model.up_proj.register_forward_hook(
        lambda _module, _inputs, output: local_hidden_sizes.append(output.shape[-1])
    )

    x_reference = torch.linspace(-1.2, 0.8, 36).reshape(2, 3, 6)
    x_reference.requires_grad_(True)
    x_parallel = x_reference.detach().clone().requires_grad_(True)
    expected = down_reference(F.gelu(up_reference(x_reference)))
    got = model(x_parallel)
    handle.remove()
    assert local_hidden_sizes == [10 // world_size]
    torch.testing.assert_close(got, expected)

    expected.square().mean().backward()
    got.square().mean().backward()
    torch.testing.assert_close(x_parallel.grad, x_reference.grad)

    reference_params = {
        "up_proj.weight": up_reference.weight,
        "up_proj.bias": up_reference.bias,
        "down_proj.weight": down_reference.weight,
        "down_proj.bias": down_reference.bias,
    }
    for name, parameter in model.named_parameters():
        expected_parameter = _tp_shard(
            name, reference_params[name], rank, world_size
        )
        torch.testing.assert_close(parameter, expected_parameter)
        torch.testing.assert_close(
            parameter.grad,
            _tp_shard(name, reference_params[name].grad, rank, world_size),
        )

    parallel_optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    reference_optimizer = torch.optim.SGD(
        [
            up_reference.weight,
            up_reference.bias,
            down_reference.weight,
            down_reference.bias,
        ],
        lr=0.05,
    )
    parallel_optimizer.step()
    reference_optimizer.step()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter,
            _tp_shard(name, reference_params[name], rank, world_size),
        )


def test_tensor_parallel_mlp_keeps_hidden_features_sharded_and_trains():
    run_gloo(2, _worker)
