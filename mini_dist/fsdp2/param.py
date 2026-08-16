from dataclasses import dataclass
from enum import Enum, auto
import torch
import torch.distributed as dist
from mini_dist.zero.common import Partition, pad_flat, partition_1d
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
    source_param: torch.nn.Parameter | None = None
    group: object | None = None

    @classmethod
    def from_parameter(cls, name: str, param: torch.nn.Parameter, *, group=None) -> "FSDPParam":
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        flat = param.detach().reshape(-1)
        partition = partition_1d(flat.numel(), rank, world_size)
        padded = pad_flat(flat, partition.padded_numel)
        local = padded[partition.start : partition.end].clone()
        return cls(
            name=name,
            original_shape=param.shape,
            partition=partition,
            local_shard=local,
            source_param_id=id(param),
            source_param=param,
            group=group,
        )

    def unshard(self, *, group=None) -> torch.Tensor:
        selected_group = self.group if group is None else group
        gathered = torch.empty(
            self.partition.padded_numel,
            dtype=self.local_shard.dtype,
            device=self.local_shard.device,
        )
        dist.all_gather_into_tensor(gathered, self.local_shard.contiguous(), group=selected_group)
        self._unsharded = gathered[: self.partition.original_numel].view(self.original_shape)
        self.state = ParamState.UNSHARDED
        if self.source_param is not None:
            with torch.no_grad():
                self.source_param.data = self._unsharded
        return self._unsharded

    def reshard(self) -> None:
        if self.source_param is not None:
            with torch.no_grad():
                self.source_param.data = self.local_shard
        self._unsharded = None
        self.state = ParamState.SHARDED
