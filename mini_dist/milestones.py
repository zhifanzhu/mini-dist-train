import torch
from ._todo import todo


def run_final_equivalence(*, world_size: int = 2, steps: int = 2) -> dict[str, torch.Tensor]:
    """Return flattened final model parameters for every implementation.

    Required keys:
      baseline, ddp, zero1, zero2, zero3, fsdp1, fsdp2

    All variants must start from identical initialization and consume data that
    corresponds to the same global batches. The returned tensors live in the
    caller process and are compared numerically by the milestone test.
    """
    todo("run_final_equivalence")
