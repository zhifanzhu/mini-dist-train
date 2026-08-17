# ex05 — ZeRO-1

Shard optimizer ownership while parameters and gradients remain replicated. Each rank updates only its owned parameter slice; updated slices must be reconstructed so every rank sees the same full parameter after `step()`.

## Required call contract

Here, “gradients remain replicated” means their **values have already been
mean-reduced and are identical across ranks** at the point where each owner
selects its update slice. The chapter test injects that already-reduced
gradient directly; it contains no hidden synchronization. In a training loop,
the engine, ex03/ex04, or the optimizer itself must arrange one equivalent
all-reduce before the local update (and must not reduce twice):

```text
local backward → mean gradient synchronization → Zero1Optimizer.step
```

Without that synchronization, each owner updates a different parameter slice
from a different rank's minibatch, and the final all-gather produces an
identical but mathematically incorrect model. Ex08 tests this integration with
different data on each rank.

## PyTorch navigation hints

- Materialize `params` as a list immediately; an optimizer normally receives a
  one-shot `module.parameters()` generator.
- Flatten parameters in stable order, pad once, and construct the real PyTorch
  optimizer over one local FP32 master `nn.Parameter`. Its state (for example,
  Adam moments) should therefore exist only for that local shard.
- Before the local optimizer step, assign the corresponding packed gradient
  slice to `local_master_param.grad`.
- Afterwards, all-gather updated master shards, remove padding, and copy the
  reconstructed values into the original parameters under `torch.no_grad()`.
  Do not replace the original `nn.Parameter` objects.
- Treat a missing original gradient as zeros so flat offsets remain aligned.


## My notes

FSDP1 tutorial is much more well-written than FSDP2.
https://docs.pytorch.org/tutorials/intermediate/FSDP1_tutorial.html

For ZeRO, reference is:
https://deepspeed.readthedocs.io/en/latest/zero3.html

Q: how will Zero1Optimizer be called? Will I learn it?
Ans: check test_ex05_zero1.py to see how this optimizer is going to be used.
- Q: is that actually how it is used? Ans: It seems so...


My Note: the original guidance here is missing some big picture.
In this exercise, step() will consume already mean-reduced, replicated gradients; in the big picture, caller will use DDP or an equivalent synchronization before calling step().