from collections.abc import Sequence
import torch
import torch.distributed as dist
from .param import FSDPParam, ParamState
from mini_dist._todo import todo


class FSDPParamGroup:
    """Parameters communicated together while retaining per-parameter identity.

    Tests intentionally accept different collective APIs. After they pass, ask
    a code-review agent to verify that communication is grouped rather than
    launched separately for every parameter.
    """

    def __init__(self, params: Sequence[FSDPParam], *, group=None):
        self.params = list(params)
        self.group = group

    @property
    def owned_parameter_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)

    @property
    def owned_parameter_ids(self) -> tuple[int, ...]:
        return tuple(p.source_param_id for p in self.params)

    def unshard(self) -> list[torch.Tensor]:
        """Use one temporary packed all-gather for the group."""
        if not self.params:
            return []
        local_sizes = [param.partition.shard_numel for param in self.params]
        offsets = []
        offset = 0
        for size in local_sizes:
            offsets.append(offset)
            offset += size
        packed_local = torch.cat([param.local_shard for param in self.params])
        world_size = dist.get_world_size(self.group)
        gathered = torch.empty(
            packed_local.numel() * world_size,
            dtype=packed_local.dtype,
            device=packed_local.device,
        )
        dist.all_gather_into_tensor(gathered, packed_local.contiguous(), group=self.group)
        gathered = gathered.view(world_size, packed_local.numel())

        fulls = []
        for param, start, size in zip(self.params, offsets, local_sizes):
            padded = torch.cat([gathered[rank, start : start + size] for rank in range(world_size)])
            full = padded[: param.partition.original_numel].view(param.original_shape)
            param._unsharded = full
            param.state = ParamState.UNSHARDED
            if param.source_param is not None:
                with torch.no_grad():
                    param.source_param.data = full
            fulls.append(full)
        return fulls

    def reshard(self) -> None:
        for param in self.params:
            param.reshard()

    def reduce_scatter_grads(self, full_grads: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Pack/pad gradients, perform grouped reduce-scatter, return local shards."""
        if len(full_grads) != len(self.params):
            raise ValueError("gradient count must match parameter count")
        if not self.params:
            return []
        world_size = dist.get_world_size(self.group)
        per_rank = [[] for _ in range(world_size)]
        for grad, param in zip(full_grads, self.params):
            flat = grad.reshape(-1)
            padded = torch.zeros(
                param.partition.padded_numel,
                dtype=flat.dtype,
                device=flat.device,
            )
            padded[: flat.numel()].copy_(flat)
            chunks = padded.chunk(world_size)
            for rank, chunk in enumerate(chunks):
                per_rank[rank].append(chunk)
        inputs = [torch.cat(parts) for parts in per_rank]
        output = torch.empty_like(inputs[0])
        dist.reduce_scatter(output, inputs, op=dist.ReduceOp.SUM, group=self.group)
        output.div_(world_size)

        shards = []
        offset = 0
        for param in self.params:
            size = param.partition.shard_numel
            shard = output[offset : offset + size].clone()
            shards.append(shard)
            offset += size
        return shards
