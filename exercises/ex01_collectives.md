# ex01 — Collective contracts

Implement the wrappers in `mini_dist/collectives.py`. Be explicit about SUM vs MEAN semantics and fixed-size reduce-scatter requirements. Reject a reduce-scatter input with `ValueError` before communication when it is not 1-D or its length is not divisible by the selected process-group size. Later DDP/ZeRO/FSDP code depends on these exact conventions.

## Files to implement

- `mini_dist/collectives.py`
