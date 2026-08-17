# ex18 — Extended: tensor-parallel MLP

Compose ex17's layers into `TensorParallelMLP`:

```text
replicated X
    ↓
column-parallel up projection
    ↓  feature-sharded activation
local GELU
    ↓  feature-sharded activation
row-parallel down projection
    ↓
replicated output
```

This is the core MLP plan used by Megatron-style implementations and by
PyTorch's tensor-parallel API. PyTorch expresses it as
`ColwiseParallel()` followed by `RowwiseParallel()`; JAX expresses the same
layout transitions using sharding specs and a reduction of partial matrix
products.

References:

- [PyTorch large-scale TP tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)
- [PyTorch tensor-parallel styles](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [JAX manual parallelism with `shard_map`](https://docs.jax.dev/en/latest/notebooks/shard_map.html#way-tensor-parallelism-tp)

## Required invariants

- `up_proj.out_features == down_proj.in_features`.
- The hidden dimension is divisible by the TP group size.
- `up_proj` returns its local output shard; do not gather it before the
  activation or down projection.
- GELU and other elementwise activations run locally without communication.
- `down_proj` consumes exactly the corresponding feature shard and returns a
  replicated output.
- The model retains only TP-local weight shards (the down-projection bias is
  replicated, as in ex17).
- Forward values, input gradients, local parameter gradients, and an optimizer
  update must equal the appropriate shards of an unsharded reference.

If the output is numerically correct but `x.grad` is missing contributions,
trace backward from the row reduction through the column projection. Ask which
layout is `Partial` at each point; the ex17 forward/backward table determines
the required communication.

Run:

```bash
pytest tests/exercises/test_ex18_tensor_parallel_mlp.py -q
```

Attention TP, sequence parallelism, and sharded cross-entropy are valuable
follow-up labs, but are intentionally not required here.
