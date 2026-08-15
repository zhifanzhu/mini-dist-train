import pytest
import torch
import torch.distributed as dist
from mini_dist.gpu_labs import run_two_gpu_nccl_probe

pytestmark = [pytest.mark.gpu, pytest.mark.nccl]


def test_nccl_collective_contracts():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires at least 2 CUDA GPUs")
    assert dist.is_nccl_available(), "PyTorch build does not expose NCCL"
    probe = run_two_gpu_nccl_probe()
    assert probe.world_size == 2
    assert probe.all_reduce_ok
    assert probe.all_gather_ok
    assert probe.reduce_scatter_ok
