# ex11 — FSDP2-style per-parameter state

Do not use a permanent FlatParameter. Each `FSDPParam` preserves original parameter identity while switching between sharded and unsharded states. Padding is still required when dim-0/numel does not divide evenly.
