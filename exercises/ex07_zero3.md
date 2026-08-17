# ex07 — ZeRO-3

Add parameter sharding. Implement flatten/pad/shard metadata for the generic ZeRO-3 core and all-gather parameters around computation. This is the common algorithmic substrate for later FSDP-style realizations.


Q: what about backward() pass? I guess it is omitted for educational purpose?
Will my operation require additional backward() impl?