# ex12 — FSDP2-style parameter groups

Group multiple `FSDPParam` objects for communication. Pack shards into a temporary collective buffer, all-gather once, then restore individual parameter views. Backward performs grouped reduce-scatter and returns rank-local shards of the mean gradient.

The tests check reconstructed values, gradient shards, padding, and state transitions without requiring a particular collective API. After they pass, ask Codex or another code-review agent to verify that parameters are temporarily packed for one grouped all-gather and one grouped reduce-scatter rather than communicated one by one. This grouped-communication review is part of ex12.

## Packing-layout hint

A single all-gather of each rank's packed local shards produces **rank-major**
storage:

```text
rank 0: [p0 shard 0 | p1 shard 0 | ...]
rank 1: [p0 shard 1 | p1 shard 1 | ...]
...
```

To rebuild one parameter, take that parameter's segment from every rank row,
concatenate those segments, then trim its own padding and restore its shape.
Simply flattening the gathered result and splitting by full parameter sizes
mixes parameter and rank order.

The grouped reduce-scatter is the inverse layout problem: for each destination
rank, concatenate that destination's padded gradient chunk from every
parameter. One reduce-scatter then returns a packed local result, which is
split back into per-parameter shards using the recorded local offsets. Divide
the SUM by group size to preserve the course's mean-gradient convention.
