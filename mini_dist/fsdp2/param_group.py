from collections.abc import Sequence
import torch
from .param import FSDPParam
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
        todo("FSDPParamGroup.unshard")

    def reshard(self) -> None:
        todo("FSDPParamGroup.reshard")

    def reduce_scatter_grads(self, full_grads: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Pack/pad gradients, perform grouped reduce-scatter, return local shards."""
        todo("FSDPParamGroup.reduce_scatter_grads")
