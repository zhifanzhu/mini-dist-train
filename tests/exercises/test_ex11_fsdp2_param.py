import torch
from mini_dist.fsdp2 import FSDPParam, ParamState
from tests._dist_test_utils import run_gloo


def _worker(rank, world_size):
    full = torch.nn.Parameter(torch.arange(13, dtype=torch.float32).reshape(13, 1))
    reference = full.detach().clone()
    source_id = id(full)
    p = FSDPParam.from_parameter("weight", full)
    assert p.source_param_id == source_id
    assert p.state is ParamState.SHARDED
    assert p.partition.original_numel == 13
    gathered = p.unshard()
    assert p.state is ParamState.UNSHARDED
    torch.testing.assert_close(gathered, reference)
    p.reshard()
    assert p.state is ParamState.SHARDED
    assert p._unsharded is None


def test_fsdp2_param_state_machine_and_padding():
    run_gloo(4, _worker)
