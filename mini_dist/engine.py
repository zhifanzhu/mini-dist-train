import torch
from torch import distributed as dist
from .collectives import all_reduce_mean
from mini_dist.zero import (
    Zero1Optimizer, Zero2Optimizer, ShardedTensor1D, MiniZeRO3
)
from ._todo import todo


class MiniDeepSpeedEngine:
    """Engine-oriented facade over the ZeRO implementations."""

    def __init__(self, module, optimizer_cls, *, zero_stage: int, group=None, **optim_kwargs):
        self.zero_stage = zero_stage
        self.module = module
        self.group = group
        if self.zero_stage == 1:
            self.optimizer = Zero1Optimizer(
                module.parameters(), optimizer_cls,
                group=group, **optim_kwargs
            )
        elif self.zero_stage == 2:
            self.optimizer = Zero2Optimizer(
                module.parameters(), optimizer_cls,
                group=group, **optim_kwargs
            )
        elif self.zero_stage == 3:
            self.zero3 = MiniZeRO3(
                module, group=group)
            # module's params are already sharded.
            self.optimizer = optimizer_cls(
                module.parameters(),
                **optim_kwargs)
        else:
            raise ValueError(f"Unknown {self.zero_stage=}")

    def __call__(self, *args, **kwargs):
        if self.zero_stage == 1:
            return self.module(*args, **kwargs)
        elif self.zero_stage == 2:
            return self.module(*args, **kwargs)
        else:
            return self.zero3(*args, **kwargs)

    def backward(self, loss: torch.Tensor) -> None:
        if self.zero_stage == 1:
            loss.backward()
        elif self.zero_stage == 2:
            loss.backward()
        else:  # 3 
            loss.backward()

    def step(self) -> None:
        if self.zero_stage == 1:
            for param in self.module.parameters():
                if param.grad is not None:
                    all_reduce_mean(param.grad, group=self.group)
            self.optimizer.step()

        elif self.zero_stage == 2:
            self.optimizer.reduce_scatter_gradients()
            self.optimizer.step()

        else:  # 3 
            self.optimizer.step()

            with torch.no_grad():
                for sp, p in zip(
                    self.zero3.shard_params, self.module.parameters()
                    ):
                    tensor = sp.all_gather(group=self.group)
                    p.data = tensor
