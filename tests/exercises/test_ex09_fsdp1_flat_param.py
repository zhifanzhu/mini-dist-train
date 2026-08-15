import torch
import torch.nn as nn
from mini_dist.fsdp1 import FlatParameterHandle
from tests._dist_test_utils import run_gloo


def _worker(rank, world_size):
    torch.manual_seed(0)
    module = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2, bias=True))
    original = {n: p.detach().clone() for n, p in module.named_parameters()}
    handle = FlatParameterHandle(module)

    infos = handle.infos
    assert {info.name for info in infos} == set(original)
    assert sum(i.numel for i in infos) == sum(p.numel() for p in original.values())
    expected_offset = 0
    for info in sorted(infos, key=lambda item: item.offset):
        ref = original[info.name]
        assert info.shape == ref.shape
        assert info.numel == ref.numel()
        assert info.offset == expected_offset
        expected_offset += ref.numel()

    assert handle.partition.original_numel == expected_offset
    assert handle.partition.padded_numel > handle.partition.original_numel
    assert handle.partition.padded_numel % world_size == 0
    assert handle.local_shard.numel() == handle.partition.shard_numel
    full_flat = handle.unshard()
    views = handle.views(full_flat)
    assert set(views) == set(original)
    for name, ref in original.items():
        assert views[name].shape == ref.shape
        torch.testing.assert_close(views[name], ref)

    first_info = min(infos, key=lambda item: item.offset)
    first = views[first_info.name].reshape(-1)
    with torch.no_grad():
        full_flat[first_info.offset] = 123
    assert first[0].item() == 123  # reconstructed parameters must be views


def test_fsdp1_flatten_pad_shard_gather_unpad_view():
    run_gloo(4, _worker)
