import torch
import torch.nn as nn
from mini_dist.zero.stage2 import Zero2Optimizer
from tests._dist_test_utils import run_gloo


def _worker(rank, world_size):
    torch.manual_seed(7)
    model = nn.Linear(8, 2, bias=False)  # 16 elements, divisible by 2
    x = torch.arange(8, dtype=torch.float32).reshape(1, 8) + rank
    model(x).sum().backward()
    local_full_grad = model.weight.grad.detach().reshape(-1).clone()

    # Build the reference mean full gradient before ZeRO-2 discards it.
    reference = local_full_grad.clone()
    torch.distributed.all_reduce(reference)
    reference /= world_size

    opt = Zero2Optimizer(model.parameters(), torch.optim.SGD, lr=0.01)
    local_grad_shards = opt.reduce_scatter_gradients()
    assert local_grad_shards is not None
    torch.testing.assert_close(opt.local_grad_shard, reference.chunk(world_size)[rank])
    assert opt.local_grad_shard.numel() == opt.partition.shard_numel
    opt.step()

    gathered = [torch.empty_like(model.weight) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, model.weight)
    for other in gathered[1:]:
        torch.testing.assert_close(other, gathered[0])


def test_zero2_retains_only_mean_gradient_partition_and_replicates_params():
    run_gloo(2, _worker)
