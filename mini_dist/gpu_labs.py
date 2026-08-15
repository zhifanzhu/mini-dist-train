from dataclasses import dataclass
from ._todo import todo


@dataclass(frozen=True)
class OverlapProbe:
    numerical_match: bool
    launched_async: bool
    used_separate_stream: bool
    waited_before_consuming: bool


def run_two_gpu_overlap_probe() -> OverlapProbe:
    """ex14 entrypoint, called only when >=2 CUDA GPUs are available."""
    todo("run_two_gpu_overlap_probe")


@dataclass(frozen=True)
class NCCLProbe:
    world_size: int
    all_reduce_ok: bool
    all_gather_ok: bool
    reduce_scatter_ok: bool


def run_two_gpu_nccl_probe() -> NCCLProbe:
    """ex15 NCCL-backed collective contract probe."""
    todo("run_two_gpu_nccl_probe")
