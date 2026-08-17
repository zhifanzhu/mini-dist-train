# ex04 — Buckets and autograd hooks

Implement bucket layout/packing and then `MiniDDP.enable_bucketed_hooks()`. The large-cap test intentionally places all gradients in one bucket: after `backward()` gradients must already be synchronized, and `num_bucket_allreduces` must report one launched bucket collective. Then experiment with smaller bucket caps.

## PyTorch navigation hints

This is intentionally a framework-engineering exercise. The recommended
mechanism is `Parameter.register_post_accumulate_grad_hook()`: it runs after
autograd has accumulated the current leaf gradient into `parameter.grad`, and
its callback receives the parameter. A regular `register_hook()` instead sees
the incoming gradient before accumulation and has different return semantics;
it can work, but requires more careful handling.

Suggested lifecycle:

1. Build deterministic buckets once from trainable parameters and store
   per-bucket readiness state.
2. Register one post-accumulate hook per parameter. Retain the returned
   `RemovableHandle`s so hook mode can be managed explicitly.
3. When every parameter in one bucket is ready, pack their `.grad` tensors,
   mean-all-reduce the flat buffer, and copy its unpacked views back into the
   original gradient tensors.
4. Clear only that bucket's readiness state for the next backward and increment
   `num_bucket_allreduces` exactly once.

Python closures are part of the trap: a hook created in a loop must capture
that iteration's bucket/parameter (use a hook factory or default arguments),
not the loop variables' final values.

`torch.cat()` necessarily copies separate gradient storages into a contiguous
communication buffer; `unpack_views()` should use slicing/`narrow()` plus
`view()` so its results alias that flat buffer. This chapter uses blocking
communication, so `backward()` itself waits for each launched bucket. Work
handles, CUDA streams, and real communication/computation overlap belong to
ex14.
