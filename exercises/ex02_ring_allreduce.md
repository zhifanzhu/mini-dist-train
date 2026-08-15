# ex02 — Ring all-reduce (optional branch)

Implement `ring_all_reduce_sum()` in place using only point-to-point communication. Preserve the input tensor's storage identity and implement the standard chunked reduce-scatter and all-gather phases. This is for communication intuition; later framework code should still use production collectives via `torch.distributed`.
