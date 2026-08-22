# ex19 — Extended: two-dimensional TP + FSDP

Build a row-major two-dimensional process mesh and compose ex18 tensor parallelism with the existing FSDP2-style implementation. Tensor parallelism uses mesh rows; FSDP uses mesh columns.

## Files to implement

- `mini_dist/tensor_parallel/hybrid.py`
