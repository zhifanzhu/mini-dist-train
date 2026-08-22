# ex02b — Tree all-reduce (optional branch)

Implement `tree_all_reduce_sum()` using only point-to-point communication. First combine partial sums through a balanced tree, then disseminate the complete sum through a tree. Blocking, nonblocking, and batched P2P APIs are all valid. Support arbitrary process-group sizes, including sizes that are not powers of two.

The tests check numerical results for a five-rank default group, a single rank, and a non-contiguous subgroup. They intentionally do not intercept particular PyTorch calls or prescribe a binary-heap, binomial, or other equivalent balanced-tree layout.

After the tests pass, ask Codex or another code-review agent to verify only the structural objective: both phases use point-to-point communication over a balanced tree rather than a linear chain, per-rank gather, or hidden collective. This narrow algorithm-structure review is part of ex02b.

## Files to implement

- `mini_dist/collectives.py`

## Hints

- `torch.distributed.get_global_rank()`
