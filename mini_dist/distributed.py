from dataclasses import dataclass
import torch.distributed as dist
from ._todo import todo


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int

    @classmethod
    def from_default_group(cls) -> "DistributedContext":
        """Return rank/world-size from the initialized default process group."""
        require_initialized()
        return cls(rank=dist.get_rank(), world_size=dist.get_world_size())


def require_initialized() -> None:
    """Infrastructure helper used by later chapters; intentionally provided."""
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    if not dist.is_initialized():
        raise RuntimeError("default process group is not initialized")
