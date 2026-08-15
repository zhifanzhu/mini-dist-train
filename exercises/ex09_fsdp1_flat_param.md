# ex09 — FSDP1-style FlatParameter

Build a permanent flat parameter representation for a module. Track original shape, numel, offsets, padded numel, and shard ranges. Exercise flatten → pad → shard → all-gather → unpad → view exactly.
