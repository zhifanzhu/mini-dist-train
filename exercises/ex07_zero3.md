# ex07 — ZeRO-3

Add parameter sharding. Implement flatten/pad/shard metadata for the generic ZeRO-3 core and all-gather parameters around computation. This is the common algorithmic substrate for later FSDP-style realizations.

## Scope: do not infer module execution order

`module.modules()` and `named_modules()` describe registration hierarchy, not
runtime forward order; reused modules, branches, and functional calls make a
general static ordering impossible. This exercise does not ask you to discover
that order. `MiniZeRO3` is a coarse wrapper: all-gather all of its parameter
shards at the beginning of the wrapper's `forward()`, then call the wrapped
module normally.

## Backward and PyTorch hook hint

Backward is part of the cumulative implementation even though the direct ex07
module test focuses on forward materialization; ex08's stage-3 workflow tests
the backward/update path. Register a
`Parameter.register_post_accumulate_grad_hook()` while the original parameter
is still the autograd leaf. After its full gradient is populated, pad and
reduce-scatter that gradient to the rank-local mean shard, restore sharded
parameter storage, and leave the local shard in `.grad` for the optimizer.

Preserve each original `nn.Parameter` object's identity—optimizers and hooks
refer to it. The exercise therefore requires controlled storage/view swapping
under `torch.no_grad()` rather than replacing parameters with newly constructed
objects. Later ex10/ex13 refine this coarse lifecycle into FSDP-style module
ownership and hooks.

For `ShardedTensor1D`, keep the ex05/ex06 metadata rule: flatten, pad to an
equal partition, store one cloned shard, then all-gather and trim back to the
exact original numel/shape.
