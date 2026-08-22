# ex17 — Extended: tensor-parallel linear primitives

Implement the four autograd-aware layout transitions in `mini_dist/tensor_parallel/collectives.py`, followed by `ColumnParallelLinear` and `RowParallelLinear`. Use raw process groups rather than a high-level tensor-parallel API.

## Files to implement

- `mini_dist/tensor_parallel/collectives.py`
- `mini_dist/tensor_parallel/layers.py`
