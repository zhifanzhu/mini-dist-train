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
        todo("BucketLayout.from_tensors")

    def pack(self, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        todo("BucketLayout.pack")

    def unpack_views(self, flat: torch.Tensor) -> list[torch.Tensor]:
        todo("BucketLayout.unpack_views")
