import torch.nn as nn
import torch
from .param import FSDPParam, ParamState
from .param_group import FSDPParamGroup
from mini_dist._todo import todo


def fully_shard(module: nn.Module, *, group=None, reshard_after_forward: bool = True) -> nn.Module:
    """FSDP2-inspired in-place API.

    Exercise invariants:
    - parameters already owned by a child fully-sharded module are not re-owned;
    - user-facing module identity is preserved;
    - hooks drive unshard/reshard around computation;
    - optimizer should be constructed after sharding.
    """
    if hasattr(module, "_mini_fsdp_group"):
        return module

    child_owned_ids = set()
    for child in module.modules():
        if child is module:
            continue
        child_group = getattr(child, "_mini_fsdp_group", None)
        if child_group is not None:
            child_owned_ids.update(child_group.owned_parameter_ids)

    owned = [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if id(parameter) not in child_owned_ids
    ]
    fsdp_params = [FSDPParam.from_parameter(name, parameter, group=group) for name, parameter in owned]
    param_group = FSDPParamGroup(fsdp_params, group=group)
    module._mini_fsdp_group = param_group
    module._mini_fsdp_reshard_after_forward = reshard_after_forward

    for fsdp_param in fsdp_params:
        with torch.no_grad():
            fsdp_param.source_param.data = fsdp_param.local_shard
            fsdp_param.local_shard = fsdp_param.source_param.data

    readiness = set()

    def pre_forward(_module, _args):
        readiness.clear()
        param_group.unshard()

    module.register_forward_pre_hook(pre_forward)

    for fsdp_param in fsdp_params:
        source = fsdp_param.source_param

        def post_accumulate(ready_parameter, *, fsdp_param=fsdp_param):
            readiness.add(id(ready_parameter))
            if len(readiness) != len(fsdp_params):
                return
            full_grads = [item.source_param.grad for item in fsdp_params]
            local_grads = param_group.reduce_scatter_grads(full_grads)
            for item, local_grad in zip(fsdp_params, local_grads):
                with torch.no_grad():
                    item.source_param.data = item.local_shard
                item.source_param.grad = local_grad.to(item.source_param.dtype)
                item.state = ParamState.SHARDED
                item._unsharded = None

        source.register_post_accumulate_grad_hook(post_accumulate)

    return module
