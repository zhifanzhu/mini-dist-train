from dataclasses import dataclass
from enum import Enum, auto
import torch
from mini_dist.zero.common import Partition
from mini_dist._todo import todo


class ParamState(Enum):
    SHARDED = auto()
    UNSHARDED = auto()


@dataclass
class FSDPParam:
    """FSDP2-style per-parameter identity/state."""
    name: str
    original_shape: torch.Size
    partition: Partition
    local_shard: torch.Tensor
    source_param_id: int
    state: ParamState = ParamState.SHARDED
    _unsharded: torch.Tensor | None = None

    @classmethod
    def from_parameter(cls, name: str, param: torch.nn.Parameter, *, group=None) -> "FSDPParam":
        todo("FSDPParam.from_parameter")

    def unshard(self, *, group=None) -> torch.Tensor:
        todo("FSDPParam.unshard")

    def reshard(self) -> None:
        todo("FSDPParam.reshard")
