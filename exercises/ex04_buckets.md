# ex04 — Buckets and autograd hooks

Implement bucket layout/packing and then `MiniDDP.enable_bucketed_hooks()`. The large-cap test intentionally places all gradients in one bucket: after `backward()` gradients must already be synchronized, and `num_bucket_allreduces` must report one launched bucket collective. Then experiment with smaller bucket caps.

- My note: this is a hard exercise.
   - edit: Codex added the below hints after I passed this. Hopefully this one is easier now.

## Files to implement

- `mini_dist/buckets.py`
- `mini_dist/ddp.py`

## Hints

- `Parameter.register_post_accumulate_grad_hook()`
