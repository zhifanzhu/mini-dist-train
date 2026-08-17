# ex09 — FSDP1-style FlatParameter

Build a permanent flat parameter representation for a module. Track original shape, numel, offsets, padded numel, and shard ranges. Exercise flatten → pad → shard → all-gather → unpad → view exactly.

## PyTorch navigation hints

- `module.named_parameters()` supplies stable qualified names and the parameter
  order used for flattening. Record each tensor's starting offset before
  advancing by its numel.
- This chapter builds metadata and storage only; it does not yet replace the
  module's registered parameters. That lifecycle begins in ex10.
- `unshard()` all-gathers equal local shards and trims padding, returning one
  exact 1-D flat tensor.
- Reconstructed tensors must alias that flat storage. `Tensor.narrow()` (or a
  slice) followed by `view(original_shape)` creates a view; cloning or
  concatenating each reconstructed parameter would fail the aliasing
  invariant.
