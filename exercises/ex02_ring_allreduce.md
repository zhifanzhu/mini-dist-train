# ex02 — Ring all-reduce (optional branch)

Implement `ring_all_reduce_sum()` using only point-to-point communication. Blocking, nonblocking, or batched P2P APIs are all valid. Implement the standard chunked reduce-scatter and all-gather phases. This is for communication intuition; later framework code should still use production collectives via `torch.distributed`.

The tests check numerical results across several group shapes but intentionally do not intercept particular PyTorch P2P calls, because doing so would reject valid implementations using `send`/`recv`, `isend`/`irecv`, or batched operations. After the tests pass, ask Codex or another code-review agent to verify that the implementation really has the two ring phases and communicates chunks rather than full tensors. This algorithm-structure review is part of ex02.
