# ex18 — Extended: tensor-parallel MLP

Compose ex17's column-parallel and row-parallel layers into `TensorParallelMLP`. Keep the intermediate feature activation sharded and return a replicated output.

## Files to implement

- `mini_dist/tensor_parallel/mlp.py`
