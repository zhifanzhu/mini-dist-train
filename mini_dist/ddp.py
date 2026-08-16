import torch
import torch.nn as nn
from .collectives import all_reduce_mean
from .buckets import BucketLayout, TensorSlice
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
                syncer.mark_ready(offset, grad)
            return hook

        # Q: shouldn't we call async_op=True here? and in ex01?
        named_params = []
        cum_numel = 0
        for name, parameter in reversed(list(self.module.named_parameters())):
            if not parameter.requires_grad:
                continue
            numel = parameter.numel()
            if cum_numel + numel >= bucket_cap_numel:
                params = [p for (n, p) in named_params]
                bucket = BucketLayout.from_tensors(params)
                syncer = BucketSyncer(bucket, named_params, self)
                for s, param in zip(bucket.slices, params):
                    offset = s.offset
                    param.register_hook(make_hook(syncer, offset))
                self._handles.append(syncer)
                named_params = []
            else:
                cum_numel += numel
                named_params.append((name, parameter))
        
        # register the rest bucket
        if len(named_params) > 0:
            params = [p for (n, p) in named_params]
            bucket = BucketLayout.from_tensors(params)
            syncer = BucketSyncer(bucket, named_params, self)
            for s, param in zip(bucket.slices, params):
                offset = s.offset
                def hook(grad):
                    syncer.mark_ready(offset, grad)
                param.register_hook(hook)
            self._handles.append(syncer)
        for syncer in self._handles:
            print(syncer.bucket.total_numel)
        print(self._handles)

        
class BucketSyncer:
    def __init__(self, bucket: BucketLayout, params: list, ddp: MiniDDP):
        """ params are assumed to be correctly offset in bucket """
        self.bucket = bucket
        self.all_offsets = set(s.offset for s in bucket.slices)
        self.params = dict()
        for s, param in zip(self.bucket.slices, params):
            self.params[s.offset] = param
        self.grads = dict()
        self.ddp = ddp  # to increment launcher
    def mark_ready(self, offset, grad):
        # this will be called inside a hook
        self.grads[offset] = grad
        print('ready', self.grads.keys(), self.all_offsets)
        if len(self.grads) == len(self.all_offsets) \
            and set(self.grads) == self.all_offsets:
            self.sync()
    def sync(self):
        print("Sync called")
        for n, p in self.params.values():
            print(f"Before sync, {n}.grad = {p.grad}")

        # sync and clear ready bits
        grad_tensors = []
        for s in self.bucket.slices:
            grad_tensors.append( self.grads[s.offset] )
        mean_grad_tensors = self.bucket.pack(grad_tensors)
        all_reduce_mean(mean_grad_tensors)
        self.grads.clear()
        views = self.bucket.unpack_views(mean_grad_tensors)
        for s, tensor_view in zip(self.bucket.slices, views):
            # self.params[s.offset].grad.copy_(tensor_view)
            n, p = self.params[s.offset]
            if p.grad is None:
                print(f"{n=} does not need grad?")
            else:
                self.params[s.offset][1].grad.copy_(tensor_view)
        self.ddp.num_bucket_allreduces += 1

        for n, p in self.params.values():
            print(f"After sync, {n}.grad = {p.grad}")