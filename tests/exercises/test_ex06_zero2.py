import copy
import torch
import torch.nn as nn
from mini_dist.zero.stage2 import Zero2Optimizer
from tests._dist_test_utils import run_gloo


def _worker(rank, world_size):
    torch.manual_seed(7)
    model = nn.Linear(8, 2, bias=True)  # 18 elements: deliberately non-divisible by 4
    reference_model = copy.deepcopy(model)
    x = torch.arange(8, dtype=torch.float32).reshape(1, 8) + rank
    model(x).sum().backward()
    reference_model(x).sum().backward()

    # Build and apply the ordinary mean-gradient reference update.
    for p in reference_model.parameters():
        torch.distributed.all_reduce(p.grad)
        p.grad.div_(world_size)
    reference_opt = torch.optim.SGD(reference_model.parameters(), lr=0.01)

    opt = Zero2Optimizer(model.parameters(), torch.optim.SGD, lr=0.01)
    local_grad_shards = opt.reduce_scatter_gradients()
    assert local_grad_shards is not None

    reference_flat_grad = torch.cat([p.grad.reshape(-1) for p in reference_model.parameters()])
    padded_reference = torch.zeros(opt.partition.padded_numel, dtype=reference_flat_grad.dtype)
    padded_reference[: reference_flat_grad.numel()] = reference_flat_grad
    torch.testing.assert_close(opt.local_grad_shard, padded_reference.chunk(world_size)[rank])
    assert opt.local_grad_shard.numel() == opt.partition.shard_numel

    reference_opt.step()
    opt.step()

    for p, expected in zip(model.parameters(), reference_model.parameters()):
        torch.testing.assert_close(p, expected)
        gathered = [torch.empty_like(p) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, p)
        for other in gathered[1:]:
            torch.testing.assert_close(other, gathered[0])


def test_zero2_retains_only_mean_gradient_partition_and_replicates_params():
    run_gloo(4, _worker)
