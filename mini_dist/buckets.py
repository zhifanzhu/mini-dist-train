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
        total_numel = 0
        slices = []
        for ten in tensors:
            numel = ten.numel()
            slices.append( TensorSlice(total_numel, numel, ten.shape) )
            total_numel += numel
        return BucketLayout(slices, total_numel)

    def pack(self, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        out = []
        for s, ten in zip(self.slices, tensors):
            assert s.shape == ten.shape
            out.append( ten.flatten() )
        # Q: Is data copying here desired?
        return torch.cat(out)

    def unpack_views(self, flat: torch.Tensor) -> list[torch.Tensor]:
        out = []
        for tenslice in self.slices:
            offset = tenslice.offset
            numel = tenslice.numel
            ten = flat[offset:offset+numel].view(tenslice.shape)
            out.append(ten)
        return out