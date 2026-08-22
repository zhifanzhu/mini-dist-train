# ex03 — MiniDDP

Implement replicated-data-parallel gradient synchronization. Start with an explicit `sync_gradients()` after backward. Keep the convention that synchronized gradients equal the mean across ranks.

## Files to implement

- `mini_dist/ddp.py`
