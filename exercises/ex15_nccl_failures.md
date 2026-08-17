# ex15 — NCCL and failure propagation

Run the same collective contracts on ProcessGroupNCCL, inspect NCCL/PyTorch debug output, and execute the scripts under `fault_injection/`. For every hang/failure, write the ordered collective sequence per rank and identify the first divergence.

## PyTorch/NCCL navigation hints

- The two-GPU probe follows one process per device: set the CUDA device before
  `init_process_group("nccl")`, and create every collective tensor on that
  rank's CUDA device. CPU tensors are not valid ProcessGroupNCCL payloads.
- Allocate collective outputs explicitly and verify all ranks. Reduce local
  boolean checks with `ReduceOp.MIN` before rank 0 reports success, so one
  rank's failure cannot be hidden.
- Launch a fault script with
  `torchrun --standalone --nproc-per-node=2 fault_injection/<case>.py`. The
  provided scripts run on Gloo/CPU by default and already use short
  process-group timeouts; a timeout is an observation, not a harness failure.
  To adapt one to NCCL, first set one CUDA device per rank and move every
  collective payload to that device—changing only the backend is insufficient.
- For a skipped or mismatched collective, compare ordered sequences—not just
  source lines. The first operation index at which ranks call different
  collectives, shapes, or no collective explains the downstream timeout.

The optional `nccl/direct_allreduce.cu` lab is separate from the Python probe.
It requires CUDA/NCCL development headers and asks you to distinguish host
enqueue from CUDA-stream completion; it is not required on a CPU-only machine.
