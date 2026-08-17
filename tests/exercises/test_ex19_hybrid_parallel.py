import copy

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from mini_dist.fsdp2 import ParamState
from mini_dist.tensor_parallel import HybridTensorFSDPMLP, ParallelMesh2D
from tests._dist_test_utils import run_gloo


def _tp_shard(name, tensor, tp_rank, tp_size):
    if name in {"up_proj.weight", "up_proj.bias"}:
        return tensor.chunk(tp_size, dim=0)[tp_rank]
    if name == "down_proj.weight":
        return tensor.chunk(tp_size, dim=1)[tp_rank]
    if name == "down_proj.bias":
        return tensor
    raise AssertionError(f"unexpected parameter {name}")


def _dp_local_shard(tensor, partition):
    padded = torch.zeros(
        partition.padded_numel, dtype=tensor.dtype, device=tensor.device
    )
    padded[: tensor.numel()].copy_(tensor.reshape(-1))
    return padded[partition.start : partition.end]


def _worker(rank, world_size):
    mesh = ParallelMesh2D(dp_size=2, tp_size=2)
    assert mesh.coordinate == (rank // 2, rank % 2)
    assert mesh.dp_rank == rank // 2
    assert mesh.tp_rank == rank % 2

    rank_tensor = torch.tensor([rank])
    tp_members = [torch.empty_like(rank_tensor) for _ in range(mesh.tp_size)]
    dp_members = [torch.empty_like(rank_tensor) for _ in range(mesh.dp_size)]
    dist.all_gather(tp_members, rank_tensor, group=mesh.tp_group)
    dist.all_gather(dp_members, rank_tensor, group=mesh.dp_group)
    assert [item.item() for item in tp_members] == [
        mesh.dp_rank * mesh.tp_size + i for i in range(mesh.tp_size)
    ]
    assert [item.item() for item in dp_members] == [
        i * mesh.tp_size + mesh.tp_rank for i in range(mesh.dp_size)
    ]

    torch.manual_seed(19)
    up_source = nn.Linear(6, 10, bias=True)
    down_source = nn.Linear(10, 4, bias=True)
    up_reference = copy.deepcopy(up_source)
    down_reference = copy.deepcopy(down_source)
    model = HybridTensorFSDPMLP(
        up_source, down_source, mesh=mesh, activation=F.gelu
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)

    # A data-parallel replica supplies one batch, replicated within its TP row.
    x = torch.linspace(-1, 1, 24).reshape(2, 2, 6) + mesh.dp_rank * 0.25
    x_reference = x.clone().requires_grad_(True)
    x_parallel = x.clone().requires_grad_(True)
    expected = down_reference(F.gelu(up_reference(x_reference)))
    got = model(x_parallel)
    torch.testing.assert_close(got, expected)

    expected.square().mean().backward()
    for parameter in list(up_reference.parameters()) + list(
        down_reference.parameters()
    ):
        dist.all_reduce(parameter.grad, group=mesh.dp_group)
        parameter.grad.div_(mesh.dp_size)

    got.square().mean().backward()

    reference_params = {
        "up_proj.weight": up_reference.weight,
        "up_proj.bias": up_reference.bias,
        "down_proj.weight": down_reference.weight,
        "down_proj.bias": down_reference.bias,
    }
    local_parameters = dict(model.tp_mlp.named_parameters())
    for fsdp_param in model.fsdp_group.params:
        assert fsdp_param.state is ParamState.SHARDED
        local_parameter = local_parameters[fsdp_param.name]
        expected_tp_param = _tp_shard(
            fsdp_param.name,
            reference_params[fsdp_param.name],
            mesh.tp_rank,
            mesh.tp_size,
        )
        expected_tp_grad = _tp_shard(
            fsdp_param.name,
            reference_params[fsdp_param.name].grad,
            mesh.tp_rank,
            mesh.tp_size,
        )
        assert fsdp_param.original_shape == expected_tp_param.shape
        torch.testing.assert_close(
            local_parameter,
            _dp_local_shard(expected_tp_param.detach(), fsdp_param.partition),
        )
        torch.testing.assert_close(
            local_parameter.grad,
            _dp_local_shard(expected_tp_grad, fsdp_param.partition),
        )

    optimizer.step()
    reference_optimizer = torch.optim.SGD(
        [
            up_reference.weight,
            up_reference.bias,
            down_reference.weight,
            down_reference.bias,
        ],
        lr=0.03,
    )
    reference_optimizer.step()
    for fsdp_param in model.fsdp_group.params:
        local_parameter = local_parameters[fsdp_param.name]
        expected_tp_param = _tp_shard(
            fsdp_param.name,
            reference_params[fsdp_param.name],
            mesh.tp_rank,
            mesh.tp_size,
        )
        torch.testing.assert_close(
            local_parameter,
            _dp_local_shard(expected_tp_param.detach(), fsdp_param.partition),
        )


def test_two_dimensional_tp_fsdp_mesh_matches_data_parallel_reference():
    run_gloo(4, _worker, timeout_s=30)
