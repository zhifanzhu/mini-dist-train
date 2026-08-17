# ex03 — MiniDDP

Implement replicated-data-parallel gradient synchronization. Start with an explicit `sync_gradients()` after backward. Keep the convention that synchronized gradients equal the mean across ranks.

## Lifecycle hint

This chapter intentionally does **not** require an autograd hook. Delegate
`forward()` to the wrapped module, let ordinary `loss.backward()` populate
each leaf parameter's `.grad`, then have the caller invoke
`sync_gradients()` before `optimizer.step()`. Iterate parameters, skip
`grad is None`, and mean-all-reduce each existing gradient in place.

Autograd hooks move this synchronization into backward in ex04. Keeping ex03
explicit makes the ordering visible:

```text
forward → backward (rank-local grads) → sync_gradients → optimizer.step
```
