import torch
import torch.nn as nn
from .collectives import all_reduce_mean
from .buckets import BucketLayout
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
        if not self._hook_mode:
            for parameter in self.module.parameters():
                if parameter.grad is not None:
                    all_reduce_mean(parameter.grad)

    def enable_bucketed_hooks(self, bucket_cap_numel: int = 1_000_000) -> None:

        # Ref: https://docs.pytorch.org/docs/2.13/notes/ddp.html#internal-design
        self._hook_mode = True

        def make_hook(syncer, offset):
            def hook(grad):
                is_last, last_grad = syncer.mark_ready(offset, grad)
                if is_last:
                    return last_grad
            return hook
        
        def register_params(named_params, sync_id):
            params = [p for (n, p) in named_params]
            bucket = BucketLayout.from_tensors(params)
            syncer = BucketSyncer(bucket, named_params, self, sync_id)
            for s, param in zip(bucket.slices, params):
                offset = s.offset
                param.register_hook(make_hook(syncer, offset))
            return syncer

        # Q: shouldn't we call async_op=True here? and in ex01?
        named_params = []
        cum_numel = 0
        for name, parameter in reversed(list(self.module.named_parameters())):
            if not parameter.requires_grad:
                continue
            numel = parameter.numel()
            if cum_numel + numel > bucket_cap_numel:  # flush existing
                syncer = register_params(named_params, len(self._handles))
                self._handles.append(syncer)
                cum_numel = 0
                named_params = []

            cum_numel += numel
            named_params.append((name, parameter))
        
        # register the rest bucket
        if len(named_params) > 0:
            syncer = register_params(named_params, len(self._handles))
            self._handles.append(syncer)

        
class BucketSyncer:
    def __init__(self, bucket: BucketLayout, named_params: list, ddp: MiniDDP,
                sync_id):
        """ params are assumed to be correctly offset in bucket.
        The name in named_params, and sync_id is used for potential debugging.
        The essential ones are `bucket` and `params`
        """
        self.sync_id = sync_id
        self.bucket = bucket
        self.all_offsets = set(s.offset for s in bucket.slices)
        self.params = dict()
        for s, (n, p) in zip(self.bucket.slices, named_params):
            self.params[s.offset] = (n, p)  # (name, param)
        self.grads = dict()
        self.ddp = ddp  # to increment launcher
    def mark_ready(self, offset, grad):
        # this will be called inside a hook
        self.grads[offset] = grad
        if len(self.grads) == len(self.all_offsets) \
            and set(self.grads) == self.all_offsets:
            last_grad = self.sync(last_offset=offset)
            return True, last_grad
        else:
            return False, None  # is_last = False, last_grad = None
    def sync(self, last_offset: int):
        # sync and clear ready bits
        grad_tensors = []
        for s in self.bucket.slices:
            grad_tensors.append( self.grads[s.offset] )
        mean_grad_tensors = self.bucket.pack(grad_tensors)
        all_reduce_mean(mean_grad_tensors)
        views = self.bucket.unpack_views(mean_grad_tensors)
        last_grad = None
        for s, tensor_view in zip(self.bucket.slices, views):
            if s.offset == last_offset:
                last_grad = tensor_view
            else:
                self.params[s.offset][1].grad.copy_(tensor_view)

        self.grads.clear()
        self.ddp.num_bucket_allreduces += 1
        return last_grad


"""
Okay this feels like a crazy exercise...and a software engineering exercise.
<del>
when a param is the last tensor to trigger bucket sync(), 
inside sync() its param.grad will be None, because sync() is itself inside hook, 
while when pytorch enters the hook, it sets param.grad to None temporarily?
But param.grad is None is also an indicator that this is the last tensor in the bucket,
so we can explicitly ask hook to return grad; for other tensors, we set by param.grad.copy_
</del>

EDIT: above is not fully correct.
The best way to track the last_grad is simply record the offset that trigger's sync()

Final words:
    This is a blocking bucket sync.
    production code will include like async_op, wait etc (instead of calling my ex01 code) 
        to make sure real comm-backward overlap.
"""
