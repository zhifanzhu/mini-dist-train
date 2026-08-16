import torch
import torch.nn as nn
import torch.distributed as dist
from .buckets import BucketLayout
from .collectives import all_reduce_mean
from ._todo import todo


class MiniDDP(nn.Module):
    """Small DDP-like wrapper grown across ex03/ex04.

    ex03: implement explicit `sync_gradients()`.
    ex04: optionally register hooks/buckets to overlap synchronization.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._hook_mode = False
        self._handles = []
        self.num_bucket_allreduces = 0  # increment once per launched bucket collective in ex04

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def sync_gradients(self) -> None:
        if self._hook_mode:
            return
        for parameter in self.module.parameters():
            if parameter.grad is not None:
                all_reduce_mean(parameter.grad)

    def enable_bucketed_hooks(self, bucket_cap_numel: int = 1_000_000) -> None:
        if bucket_cap_numel <= 0:
            raise ValueError("bucket_cap_numel must be positive")
        if self._hook_mode:
            return

        parameters = [parameter for parameter in self.module.parameters() if parameter.requires_grad]
        buckets: list[list[nn.Parameter]] = []
        current: list[nn.Parameter] = []
        current_numel = 0
        for parameter in parameters:
            if current and current_numel + parameter.numel() > bucket_cap_numel:
                buckets.append(current)
                current = []
                current_numel = 0
            current.append(parameter)
            current_numel += parameter.numel()
        if current:
            buckets.append(current)

        self._hook_mode = True
        self._bucket_state = []
        for bucket in buckets:
            layout = BucketLayout.from_tensors(bucket)
            state = {"params": bucket, "layout": layout, "ready": {}}
            self._bucket_state.append(state)

            for parameter in bucket:
                def hook(ready_parameter, *, state=state):
                    state["ready"][id(ready_parameter)] = True
                    if len(state["ready"]) == len(state["params"]):
                        gradients = [item.grad for item in state["params"]]
                        flat = state["layout"].pack(gradients)
                        all_reduce_mean(flat)
                        for destination, reduced in zip(gradients, state["layout"].unpack_views(flat)):
                            destination.copy_(reduced)
                        state["ready"].clear()
                        self.num_bucket_allreduces += 1

                self._handles.append(parameter.register_post_accumulate_grad_hook(hook))
