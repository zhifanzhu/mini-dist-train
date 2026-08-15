import torch
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


def _hook_worker(rank, world_size):
    torch.manual_seed(0)
    model = MiniDDP(nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1)))
    model.enable_bucketed_hooks(bucket_cap_numel=10_000)  # all grads should fit one bucket
    x = torch.arange(4, dtype=torch.float32).reshape(1, 4) + rank
    model(x).sum().backward()

    # Hooks must have synchronized gradients without explicit sync_gradients().
    for p in model.parameters():
        if p.grad is None:
            continue
        gathered = [torch.empty_like(p.grad) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, p.grad)
        for other in gathered[1:]:
            torch.testing.assert_close(other, gathered[0])
    assert model.num_bucket_allreduces == 1


def test_bucketed_autograd_hooks_launch_when_bucket_ready():
    run_gloo(2, _hook_worker)
