import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from pathlib import Path
import tempfile
from .ddp import MiniDDP
from .fsdp1 import MiniFSDP1
from .fsdp2 import fully_shard
from .zero.stage1 import Zero1Optimizer
from .zero.stage2 import Zero2Optimizer
from ._todo import todo


def _model() -> nn.Module:
    torch.manual_seed(2026)
    return nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))


def _input(rank: int, step: int) -> torch.Tensor:
    return torch.tensor(
        [[1.0 + rank, -0.5 + step, float(rank - step)]],
        dtype=torch.float32,
    )


def _flatten(module: nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1).cpu() for parameter in module.parameters()])


def _reference(world_size: int, steps: int) -> torch.Tensor:
    module = _model()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.03)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        for rank in range(world_size):
            module(_input(rank, step)).sum().div(world_size).backward()
        optimizer.step()
    return _flatten(module)


def _distributed_worker(rank, world_size, init_file, result_file, variant, steps):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        module = _model()
        final_parameters = None

        if variant == "ddp":
            wrapped = MiniDDP(module)
            optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.03)
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                wrapped(_input(rank, step)).sum().backward()
                wrapped.sync_gradients()
                optimizer.step()

        elif variant == "zero1":
            optimizer = Zero1Optimizer(module.parameters(), torch.optim.SGD, lr=0.03)
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                module(_input(rank, step)).sum().backward()
                for parameter in module.parameters():
                    parameter.grad.div_(world_size)
                    dist.all_reduce(parameter.grad)
                optimizer.step()

        elif variant == "zero2":
            optimizer = Zero2Optimizer(module.parameters(), torch.optim.SGD, lr=0.03)
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                module(_input(rank, step)).sum().backward()
                optimizer.step()

        elif variant == "fsdp1":
            wrapped = MiniFSDP1(module)
            optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.03)
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                wrapped(_input(rank, step)).sum().backward()
                optimizer.step()
            wrapped.handle._local_shard = wrapped.flat_param
            final_parameters = wrapped.handle.unshard().detach().cpu()

        elif variant in ("zero3", "fsdp2"):
            fully_shard(module)
            optimizer = torch.optim.SGD(module.parameters(), lr=0.03)
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                module(_input(rank, step)).sum().backward()
                optimizer.step()
            module._mini_fsdp_group.unshard()

        else:
            raise ValueError(variant)

        if rank == 0:
            torch.save(final_parameters if final_parameters is not None else _flatten(module), result_file)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def run_final_equivalence(*, world_size: int = 2, steps: int = 2) -> dict[str, torch.Tensor]:
    """Return flattened final model parameters for every implementation.

    Required keys:
      baseline, ddp, zero1, zero2, zero3, fsdp1, fsdp2

    All variants must start from identical initialization and consume data that
    corresponds to the same global batches. The returned tensors live in the
    caller process and are compared numerically by the milestone test.
    """
    if world_size <= 0 or steps <= 0:
        raise ValueError("world_size and steps must be positive")
    results = {"baseline": _reference(world_size, steps)}
    for variant in ("ddp", "zero1", "zero2", "zero3", "fsdp1", "fsdp2"):
        with tempfile.TemporaryDirectory() as directory:
            init_file = str(Path(directory) / "init")
            result_file = str(Path(directory) / "result.pt")
            mp.spawn(
                _distributed_worker,
                args=(world_size, init_file, result_file, variant, steps),
                nprocs=world_size,
                join=True,
            )
            results[variant] = torch.load(result_file, weights_only=True)
    return results
