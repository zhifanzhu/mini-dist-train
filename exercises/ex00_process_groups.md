# ex00 — Process groups and SPMD

Implement `DistributedContext.from_default_group()`. Learn that every rank executes the same program with rank-local state. Tests check rank/world-size correctness under Gloo.

## PyTorch navigation hint

The test harness has already called `torch.distributed.init_process_group()`;
do not initialize or destroy it inside `from_default_group()`. Check
`dist.is_available()`/`dist.is_initialized()`, then read the default group's
state with `dist.get_rank()` and `dist.get_world_size()`.
