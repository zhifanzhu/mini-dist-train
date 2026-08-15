# ex15 — NCCL and failure propagation

Run the same collective contracts on ProcessGroupNCCL, inspect NCCL/PyTorch debug output, and execute the scripts under `fault_injection/`. For every hang/failure, write the ordered collective sequence per rank and identify the first divergence.
