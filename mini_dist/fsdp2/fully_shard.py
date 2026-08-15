import torch.nn as nn
from mini_dist._todo import todo


def fully_shard(module: nn.Module, *, group=None, reshard_after_forward: bool = True) -> nn.Module:
    """FSDP2-inspired in-place API.

    Exercise invariants:
    - parameters already owned by a child fully-sharded module are not re-owned;
    - user-facing module identity is preserved;
    - hooks drive unshard/reshard around computation;
    - optimizer should be constructed after sharding.
    """
    todo("fully_shard")
