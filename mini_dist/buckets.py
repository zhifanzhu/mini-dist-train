from dataclasses import dataclass
from typing import Iterable, Sequence
import torch
from ._todo import todo


@dataclass(frozen=True)
class TensorSlice:
    offset: int
    numel: int
    shape: torch.Size


@dataclass(frozen=True)
class BucketLayout:
    slices: tuple[TensorSlice, ...]
    total_numel: int

    @classmethod
    def from_tensors(cls, tensors: Sequence[torch.Tensor]) -> "BucketLayout":
        slices = []
        offset = 0
        for tensor in tensors:
            slices.append(TensorSlice(offset, tensor.numel(), tensor.shape))
            offset += tensor.numel()
        return cls(tuple(slices), offset)

    def pack(self, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(tensors) != len(self.slices):
            raise ValueError(f"expected {len(self.slices)} tensors, got {len(tensors)}")
        if not tensors:
            return torch.empty(0)
        for tensor, spec in zip(tensors, self.slices):
            if tensor.numel() != spec.numel:
                raise ValueError(f"expected {spec.numel} elements, got {tensor.numel()}")
        return torch.cat([tensor.reshape(-1) for tensor in tensors])

    def unpack_views(self, flat: torch.Tensor) -> list[torch.Tensor]:
        if flat.ndim != 1 or flat.numel() != self.total_numel:
            raise ValueError(f"expected flat tensor with {self.total_numel} elements")
        return [flat.narrow(0, spec.offset, spec.numel).view(spec.shape) for spec in self.slices]
