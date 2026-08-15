import pytest
import torch
from mini_dist.gpu_labs import run_two_gpu_overlap_probe

pytestmark = [pytest.mark.gpu, pytest.mark.nccl]


def test_async_overlap_preserves_correctness_and_stream_dependencies():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires at least 2 CUDA GPUs")
    probe = run_two_gpu_overlap_probe()
    assert probe.numerical_match
    assert probe.launched_async
    assert probe.used_separate_stream
    assert probe.waited_before_consuming
