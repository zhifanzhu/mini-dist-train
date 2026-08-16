# Fault-injection analysis

These scripts are intentionally incorrect; their "solution" is identifying the first divergent operation and predicting how failure propagates.

## `skipped_collective.py`

- Rank 1: `barrier`
- Every other rank: `all_reduce` → `barrier`
- First divergence: collective index 0 is `barrier` on rank 1 and `all_reduce` elsewhere.
- Result: neither collective can acquire all required participants; the process-group timeout reports the failure.

## `mismatched_collective.py`

- Rank 0: `all_reduce` → `barrier`
- Every other rank: `all_gather` → `barrier`
- First divergence: collective index 0 uses different collective types and contracts.
- Result: the backend reports a mismatch or times out; the later barrier is never reached consistently.

## `rank_exit.py`

- Rank 1: exits before the first collective.
- Every other rank: enters `all_reduce`.
- First divergence: rank 1 has no collective at index 0.
- Result: peers fail the all-reduce with a closed connection, aborted communicator, or timeout.

## `straggler.py`

- Rank 1: sleeps for 12 seconds → `all_reduce`.
- Every other rank: immediately enters `all_reduce`.
- The collective order agrees, but the 12-second delay exceeds the configured 8-second process-group timeout.
- Result: early ranks time out; when rank 1 eventually arrives, the communicator has already failed.

## `pre_collective_exception.py`

- Rank 1: raises before the collective.
- Every other rank: enters `reduce_scatter_tensor`.
- First divergence: rank 1 has no collective at index 0.
- Result: rank 1 reports the original exception; peers report connection loss, communicator abort, or timeout from the reduce-scatter.
