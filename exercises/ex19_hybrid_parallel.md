# ex19 — Extended: two-dimensional TP + FSDP

Build a row-major two-dimensional process mesh and compose ex18 tensor
parallelism with the existing FSDP2-style implementation.

For `dp_size=2`, `tp_size=2`, global ranks are arranged as:

```text
                         TP dimension
                    tp=0             tp=1
DP dimension  dp=0   rank 0 ───────── rank 1
              dp=1   rank 2 ───────── rank 3
                       │                 │
                       │                 │
                 DP group [0,2]    DP group [1,3]

TP groups: [0,1], [2,3]
```

This mirrors PyTorch `DeviceMesh` dimensions named `("dp", "tp")` and JAX
meshes with separate data and feature axes. Production code passes the TP
submesh to tensor parallelism and the DP submesh to FSDP.

References:

- [PyTorch `DeviceMesh`](https://docs.pytorch.org/docs/stable/distributed.device_mesh.html)
- [PyTorch FSDP2 `fully_shard`](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
- [PyTorch DTensor design](https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/README.md)
- [JAX mesh and `shard_map`](https://docs.jax.dev/en/latest/notebooks/shard_map.html)

## Part 1: `ParallelMesh2D`

- Require `world_size == dp_size * tp_size` with positive dimensions.
- Use row-major coordinates:
  `global_rank = dp_rank * tp_size + tp_rank`.
- Every process must call `dist.new_group` for *all* TP and DP groups in the
  same global order, retaining only its current row and column. Creating only
  the current rank's groups can hang because process-group creation itself is
  globally ordered.
- Expose `dp_group`, `tp_group`, `dp_rank`, `tp_rank`, sizes, and
  `coordinate == (dp_rank, tp_rank)`.

## Part 2: TP-local parameters become FSDP logical parameters

Construct `TensorParallelMLP` over `mesh.tp_group` first. Each rank now owns a
TP-local weight shard. Across a fixed TP coordinate—the DP column—those local
weights are replicas. Apply the existing `fully_shard` over `mesh.dp_group`:

```text
full model parameter
    └── TP shard across a mesh row
           └── FSDP shard across the matching mesh column
```

Before a matrix multiplication, FSDP all-gathers only the TP-local parameter
within the DP group. TP computation and its partial reductions then run across
the TP group. Backward reduce-scatters each TP-local gradient over the DP
group, producing the mean across data-parallel replicas.

Important ordering hints:

- Inputs differ between DP rows but must be identical inside each TP row.
- Construct the optimizer only after FSDP has replaced parameter storage with
  local shards.
- All ranks must enter TP and DP collectives in the same per-group order.
- From FSDP's point of view, the TP-local tensor—not the original global
  tensor—is the `original_shape` being sharded.

Run:

```bash
pytest tests/exercises/test_ex19_hybrid_parallel.py -q
```

The four-rank Gloo test checks mesh membership, TP+FSDP ownership, forward
equivalence for different DP batches, mean local gradients, sharded state, and
an optimizer update. Replicated DDP instead of FSDP uses the same mesh columns
but all-reduces the TP-local gradients rather than reduce-scattering them.

This exercise intentionally stops at a 2D default-world mesh. Pipeline,
context, expert, sequence, and loss-parallel dimensions are outside scope.
