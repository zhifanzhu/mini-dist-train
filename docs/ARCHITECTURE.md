# Architecture map: algorithm vs framework realization

This course deliberately separates **the sharding algorithm** from **the framework surface**.

## The shared data-parallel progression

```text
DDP
  replicated parameters
  replicated reduced gradients
  replicated optimizer state

        ↓ shard optimizer state

ZeRO-1
  replicated parameters
  replicated reduced gradients
  sharded optimizer state

        ↓ shard reduced gradients

ZeRO-2
  replicated parameters
  sharded reduced gradients
  sharded optimizer state

        ↓ shard parameters

ZeRO-3
  sharded parameters at rest
  sharded reduced gradients
  sharded optimizer state
  all-gather parameters around computation
```

The course implements this progression before framework-specific realizations so that FSDP does not appear as a magical independent algorithm.

## DeepSpeed-style realization

`MiniDeepSpeedEngine` owns more of the training workflow:

```text
engine(x)
engine.backward(loss)
engine.step()
```

The point of ex08 is primarily **abstraction ownership**: the same ZeRO mechanisms can be exposed through an engine that coordinates backward/step.

## FSDP1-style realization

The educational FSDP1 path uses a permanent flattened parameter representation:

```text
original parameters
   ↓ flatten + metadata
FlatParameter-like buffer
   ↓ pad + shard
rank-local flat shard
```

Before computation:

```text
local flat shard
   ↓ all-gather
full padded flat buffer
   ↓ unpad
full exact flat buffer
   ↓ views using offsets/shapes
original logical parameters
```

This makes offsets, storage aliasing, padding, and parameter replacement explicit.

## FSDP2-style realization

The educational FSDP2 path preserves per-parameter identity:

```text
FSDPParam(weight)
FSDPParam(bias)
FSDPParam(...)
```

Each parameter has its own shard metadata/state. Parameters may still be **temporarily packed together for communication**:

```text
local param shards
   ↓ temporary pack
one communication buffer
   ↓ all-gather
unpack into individual full parameter views
```

The distinction to learn is:

- FSDP1-style: flattening is part of the persistent parameter representation;
- FSDP2-style: individual parameters remain first-class; packing is a communication optimization.

## Communication substrate

```text
MiniDDP / ZeRO / MiniFSDP
          ↓
torch.distributed collectives
          ↓
ProcessGroup (Gloo or NCCL)
          ↓
backend implementation
```

The ring all-reduce exercise branches below the framework layer only to teach the mechanics of constructing a collective from point-to-point operations. The framework itself should delegate to production collectives.

## State and ownership are first-class

For every exercise, be able to answer these questions at a particular point in time:

1. Does this rank own the full parameter or only a shard?
2. Does this rank own the full gradient or only a shard?
3. Which optimizer-state shard does this rank own?
4. Is the full parameter merely materialized temporarily?
5. Which collective must every rank enter next, and in what order?
6. What metadata is needed to recover the exact original parameter after padding?

If those answers are unclear, do not move on just because a numerical test happens to pass.
