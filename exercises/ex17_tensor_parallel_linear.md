# ex17 — Extended: tensor-parallel linear primitives

Implement the four autograd-aware layout transitions in
`mini_dist/tensor_parallel/collectives.py`, followed by
`ColumnParallelLinear` and `RowParallelLinear` in `layers.py`.

This exercise deliberately exposes the local SPMD mechanics hidden by
PyTorch DTensor and JAX sharding propagation. It uses raw process groups so
you must state both the forward layout transition and its backward dual.

## Production correspondence

- PyTorch [`ColwiseParallel` and `RowwiseParallel`](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html)
  transform existing `nn.Linear` modules and track `Shard`, `Replicate`, and
  `Partial` DTensor placements.
- The [PyTorch tensor-parallel tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)
  composes a column-wise projection directly with a row-wise projection.
- JAX [`shard_map`](https://docs.jax.dev/en/latest/notebooks/shard_map.html)
  exposes the same per-device view using named mesh axes and collectives such
  as `psum`/`psum_scatter`.

Do not solve the exercise by calling PyTorch's high-level tensor-parallel API;
the point is to reconstruct the layer beneath that API. Calling
`torch.distributed` collectives is expected.

## Linear layout algebra

PyTorch stores a linear weight as `W[out_features, in_features]` and computes
`Y = X Wᵀ + b`.

### Column parallel

Despite the name, this shards weight dimension 0 in PyTorch's storage layout:

```text
W = [ W0 ]       Y = [Y0 | Y1 | ...]
    [ W1 ]       Yr = X Wrᵀ + br
    [ .. ]
```

Every rank receives replicated `X` and owns one contiguous output-feature
shard. Forward needs no reduction. During backward, each rank computes only a
partial contribution to `dX`, so those contributions must be SUM-reduced.

### Row parallel

This shards weight dimension 1 and consumes the matching activation shard:

```text
W = [W0 | W1 | ...]    X = [X0 | X1 | ...]
Y = Σr Xr Wrᵀ + b
```

Each local matrix multiplication produces a partial output. SUM the partials,
then add the replicated bias exactly once. Backward sends the replicated
output gradient unchanged into every local matrix multiplication.

## Autograd-aware collective hints

The following table is the central invariant. A raw in-place collective does
not automatically give these backward semantics.

| Operation | Forward | Backward |
|---|---|---|
| copy to TP region | identity | all-reduce SUM |
| reduce from TP region | all-reduce SUM | identity |
| scatter to TP region | split last dimension | all-gather last dimension |
| gather from TP region | all-gather last dimension | split last dimension |

Implementing each mapping as a small `torch.autograd.Function` is the most
direct route. Store the process group on `ctx`. Avoid mutating a tensor whose
producer may need its original value: clone/contiguous-copy before an in-place
collective when necessary.

Additional invariants:

- `dist.get_rank(group)` is the rank *inside that group* and selects the local
  contiguous shard.
- Reject a non-divisible feature dimension with `ValueError` before entering a
  collective.
- The transformed module owns only local `nn.Parameter` objects. Do not retain
  the source `nn.Linear` as a child module.
- Preserve dtype, device, arbitrary leading batch dimensions, and optional
  bias.

Run:

```bash
pytest tests/exercises/test_ex17_tensor_parallel_linear.py -q
```

The tests cover collective forward/backward duals, local ownership, gathered
and sharded column outputs, row partial reduction, input/weight/bias gradients,
and divisibility errors on two Gloo ranks.

Equal feature partitioning is intentional for this extended lab. Uneven TP
feature shards, vocabulary parallel loss, and mixed-dtype communication are
outside its required scope.
