# ex11 — FSDP2-style per-parameter state

Do not use a permanent FlatParameter. Each `FSDPParam` preserves original parameter identity while switching between sharded and unsharded states. Padding is still required when dim-0/numel does not divide evenly.

## State and storage hints

`FSDPParam` is a state object around one existing `nn.Parameter`, not a new
module parameter. Record the source object's identity, exact shape, partition
metadata, selected process group, and one cloned local flat shard.

`unshard()` all-gathers and trims the padded tensor, changes state to
`UNSHARDED`, and makes the original parameter expose that full-shaped storage.
`reshard()` restores the same parameter object's local storage, clears the
temporary full tensor, and changes state back to `SHARDED`. As in ex07, this is
a controlled framework exercise in swapping storage under `torch.no_grad()`;
do not replace the source `nn.Parameter`, because later ownership checks use
its identity.
