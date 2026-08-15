# ex04 — Buckets and autograd hooks

Implement bucket layout/packing and then `MiniDDP.enable_bucketed_hooks()`. The large-cap test intentionally places all gradients in one bucket: after `backward()` gradients must already be synchronized, and `num_bucket_allreduces` must report one launched bucket collective. Then experiment with smaller bucket caps.
