import torch
import torch.nn as nn
from mini_dist.fsdp1 import FlatParameterHandle
from tests._dist_test_utils import run_gloo


def _worker(rank, world_size):
    torch.manual_seed(0)
    module = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2, bias=False))
    original = {n: p.detach().clone() for n, p in module.named_parameters()}
    handle = FlatParameterHandle(module)

    infos = handle.infos
    assert sum(i.numel for i in infos) == sum(p.numel() for p in original.values())
    assert handle.partition.padded_numel % world_size == 0
    full_flat = handle.unshard()
    views = handle.views(full_flat)
    assert set(views) == set(original)
    for name, ref in original.items():
        assert views[name].shape == ref.shape
        torch.testing.assert_close(views[name], ref)


def test_fsdp1_flatten_pad_shard_gather_unpad_view():
    run_gloo(4, _worker)
