import copy
import torch
import torch.distributed as dist
import torch.nn as nn
from mini_dist.buckets import BucketLayout
from mini_dist.ddp import MiniDDP
from tests._dist_test_utils import run_gloo


def test_bucket_pack_unpack_preserves_shapes_and_aliases():
    tensors = [torch.arange(6).reshape(2, 3).float(), torch.tensor([10.0, 11.0]), torch.ones(1, 2, 2)]
    layout = BucketLayout.from_tensors(tensors)
    assert layout.total_numel == 12
    flat = layout.pack(tensors)
    assert flat.shape == (12,)
    views = layout.unpack_views(flat)
    for got, src in zip(views, tensors):
        assert got.shape == src.shape
        torch.testing.assert_close(got, src)
    flat[0] = 99
    assert views[0].reshape(-1)[0].item() == 99  # views, not copies


def _hook_worker(rank, world_size, bucket_cap_numel, expect_multiple_buckets):
    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))
    reference = copy.deepcopy(base)
    model = MiniDDP(base)
    model.enable_bucketed_hooks(bucket_cap_numel=bucket_cap_numel)
    x = torch.arange(4, dtype=torch.float32).reshape(1, 4) + rank

    reference(x).sum().backward()
    for p in reference.parameters():
        dist.all_reduce(p.grad)
        p.grad.div_(world_size)

    model(x).sum().backward()

    # Hooks must produce the reference mean gradients without an explicit sync.
    for got, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(got.grad, expected.grad)

    if expect_multiple_buckets:
        assert model.num_bucket_allreduces > 1
    else:
        assert model.num_bucket_allreduces == 1


def test_bucketed_autograd_hooks_launch_when_bucket_ready():
    run_gloo(2, _hook_worker, 10_000, False)


def test_small_bucket_cap_launches_multiple_correct_reductions():
    run_gloo(2, _hook_worker, 20, True)
