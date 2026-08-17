# ex16 — Final equivalence

Train the same small model from identical initialization/data under baseline, DDP, ZeRO stages, MiniFSDP1, and MiniFSDP2. Compare final parameters, rank-local ownership, peak memory (GPU extension), and communication counts.

## Harness and comparison hints

This milestone is an integration harness. Give every realization identical
initial parameters, optimizer hyperparameters, step count, and logical global
batches. A convenient single-process baseline loops over each rank's sample,
scales each loss by `1 / world_size`, accumulates gradients, and takes one
optimizer step; distributed variants process one sample per rank and must
produce that same mean update.

Use `torch.multiprocessing.spawn` with a fresh file-initialized Gloo process
group for each realization. Child processes cannot return tensors directly to
the parent, so rank 0 may save a final flattened CPU tensor to a temporary
result file after all ranks have completed matching communication. Preserve
exact flatten order via `module.parameters()` and compare numerical values,
not only cross-rank equality.

Before debugging a mismatch, compare in this order: initialization and data,
SUM-vs-MEAN convention, gradient ownership, parameter ownership, optimizer
state ownership, padding metadata, and the update/all-gather sequence.

The required CPU milestone test checks final numerical equivalence. Peak-memory
and communication-count comparisons are observational GPU extensions unless a
later test explicitly exposes those measurements.
