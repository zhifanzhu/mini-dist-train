import torch
from mini_dist.fsdp2 import FSDPParam, FSDPParamGroup, ParamState
from tests._dist_test_utils import run_gloo


def _worker(rank, world_size):
    originals = [
        torch.nn.Parameter(torch.arange(5, dtype=torch.float32)),
        torch.nn.Parameter(torch.arange(11, dtype=torch.float32) + 20),
        torch.nn.Parameter(torch.arange(3, dtype=torch.float32) - 4),
    ]
    params = [FSDPParam.from_parameter(f"p{i}", p) for i,p in enumerate(originals)]
    group = FSDPParamGroup(params)
    fulls = group.unshard()
    for got, ref in zip(fulls, originals):
        torch.testing.assert_close(got, ref.detach())
    assert all(p.state is ParamState.UNSHARDED for p in params)

    grads = [torch.ones_like(x) * (rank + 1) for x in fulls]
    local_grad_shards = group.reduce_scatter_grads(grads)
    assert len(local_grad_shards) == len(params)
    mean_value = sum(r + 1 for r in range(world_size)) / world_size
    for local_grad, param in zip(local_grad_shards, params):
        expected = torch.zeros(param.partition.padded_numel)
        expected[: param.partition.original_numel] = mean_value
        torch.testing.assert_close(local_grad, expected.chunk(world_size)[rank])
    group.reshard()
    assert all(p.state is ParamState.SHARDED for p in params)


def test_param_group_batches_uneven_parameters():
    run_gloo(4, _worker)
