# ex01 — Collective contracts

Implement the wrappers in `mini_dist/collectives.py`. Be explicit about SUM vs MEAN semantics and fixed-size reduce-scatter requirements. Reject a reduce-scatter input with `ValueError` before communication when it is not 1-D or its length is not divisible by the selected process-group size. Later DDP/ZeRO/FSDP code depends on these exact conventions.

## PyTorch navigation hints

- `dist.all_reduce()` mutates its input. `all_reduce_mean()` must preserve that
  storage and return the same tensor object; scale by the selected group's
  world size before or after the SUM.
- All-gather APIs require output storage to be allocated explicitly. The
  exercise wants equal-shaped inputs concatenated along dimension 0.
- The list form of `dist.reduce_scatter()` consumes one equal-sized input
  chunk per destination rank and writes only this rank's reduced chunk.
- Use `dist.get_world_size(group)` and `dist.get_rank(group)` when a group is
  supplied. A non-default group's size—not the global world size—defines the
  split.
- Validate on every rank before entering a collective. Letting one rank raise
  while its peers enter communication turns an ordinary shape error into a
  timeout.
