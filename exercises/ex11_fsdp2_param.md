# ex11 — FSDP2-style per-parameter state

Do not use a permanent FlatParameter. Each `FSDPParam` preserves original parameter identity while switching between sharded and unsharded states. Padding is still required when dim-0/numel does not divide evenly.

## Differences from production FSDP2

Production FSDP2 represents sharded parameters as `DTensor`: `param.shape` remains
global, while `param.to_local()` exposes local storage. For educational purposes,
this exercise models the lifecycle by swapping `nn.Parameter` storage.

## Files to implement

- `mini_dist/fsdp2/param.py`
