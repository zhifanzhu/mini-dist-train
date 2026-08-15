import torch
import torch.distributed as dist
from .distributed import require_initialized
from ._todo import todo


def all_reduce_mean(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """In-place mean all-reduce; return `tensor` for convenience."""
    ws = dist.get_world_size(group)
    tensor.div_(ws)
    dist.all_reduce(tensor, dist.ReduceOp.SUM, group=group)
    return tensor


def all_gather_fixed(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Gather equal-sized tensors and concatenate along dim 0."""
    ws = dist.get_world_size(group)
    dd = {'device': tensor.device, 'dtype': tensor.dtype}
    shape = tensor.shape
    # rank = dist.get_rank()
    # print(rank, shape, tensor)
    tensor_out = torch.zeros([shape[0] * ws, *shape[1:]], **dd)
    dist.all_gather_single(tensor_out, tensor, group=group)
    # print(tensor_out.shape, tensor_out)
    return tensor_out


def reduce_scatter_sum(flat_tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Split a 1-D tensor evenly and reduce-scatter SUM into one shard per rank.

    
    rank 0:     [ a0 | a1 | ... | a_k-1 ]
    rank 1:     [ b0 | b1 | ... | b_k-1 ]
    ...
    rank k-1:   [ t0 | t1 | ... | t_k-1 ]
                  ^                ^
                r0 reduce            r_k-1 reduce

    output:
        rank 0 = [ a0 + b0 + ... + t0]
        rank 1 = [ a1 + b1 + ... + b_{k-1}]
        ...
        rank k-1 = [ a_k-1 + b_k-1 + ... + t_k-1 ]
    """
    if flat_tensor.ndim != 1:
        raise ValueError(f"Expect 1-D tensor, got {flat_tensor.shape}")
    ws = dist.get_world_size(group)
    if flat_tensor.shape[0] % ws != 0:
        raise ValueError(f"Expect flat_tensor divisible by process-group size {ws}")
    split_size = flat_tensor.shape[0] // ws
    splits = list(flat_tensor.split(split_size))
    tensor_out = torch.zeros_like(splits[0])
    dist.reduce_scatter(tensor_out, splits, op=dist.ReduceOp.SUM, group=group)
    # rank = dist.get_rank()
    # print(rank, flat_tensor, splits)
    # print(rank, tensor_out)
    return tensor_out


def ring_all_reduce_sum(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Optional exercise: all-reduce SUM using point-to-point communication only.

    You may assume a 1-D tensor whose length is divisible by world_size.
    Do not call dist.all_reduce/reduce_scatter/all_gather inside this function.
    """
    # send() must be answered by recv() first, otherwise deadlock.
    if group is None:
        group = dist.GroupMember.WORLD
    ws = dist.get_world_size(group)
    rank = dist.get_rank(group)
    if tensor.shape[0] % ws != 0:
        raise ValueError(f"Tensor length ({tensor.shape[0]}) not divisible by world_size={ws}.")
    chunks = list(tensor.chunk(ws))

    buffer = torch.zeros_like(chunks[0])

    """ v3, use isend/irecv to avoid odd number tricky handling. """
    #  ranks: A B C D 
    #         d a b c
    #         c d a b
    #         b c d a
    for i in range(ws-1):
        send_chunk_ind = (rank + ws - 1 - i) % ws
        dst_rank = (rank + 1) % ws
        dst_rank = dist.get_global_rank(group, dst_rank)
        write_chunk_ind = (rank + ws - 2 - i) % ws
        src_rank = (rank + ws - 1) % ws
        src_rank = dist.get_global_rank(group, src_rank)
        rs = dist.isend(chunks[send_chunk_ind], dst_rank, group)
        rr = dist.irecv(buffer, src_rank, group)
        rr.wait()
        rs.wait()
        chunks[write_chunk_ind] += buffer

    for i in range(ws-1):
        send_chunk_ind = (rank + ws - i) % ws
        dst_rank = (rank + 1) % ws
        write_chunk_ind = (rank + ws- 1 - i) % ws
        src_rank = (rank + ws - 1) % ws
        rs = dist.isend(chunks[send_chunk_ind], dst_rank, group)
        rr = dist.irecv(chunks[write_chunk_ind], src_rank, group)
        rr.wait()
        rs.wait()
    
    return tensor

    # """ v2 """
    # #  ranks: A B C D 
    # #         d a b c
    # #         c d a b
    # #         b c d a
    # for i in range(ws-1):
    #     send_chunk_ind = (rank + ws - 1 - i) % ws
    #     dst_rank = (rank + 1) % ws
    #     write_chunk_ind = (rank + ws - 2 - i) % ws
    #     src_rank = (rank + ws - 1) % ws
    #     # print(f"{rank=}, {send_chunk_ind=}->{dst_rank=}, {src_rank=}->{write_chunk_ind=}.")
    #     if rank % 2 == 0:
    #         dist.send(chunks[send_chunk_ind], dst_rank, group)
    #         dist.recv(buffer, src_rank, group)
    #         chunks[write_chunk_ind] += buffer
    #     else:
    #         dist.recv(buffer, src_rank, group)
    #         chunks[write_chunk_ind] += buffer
    #         dist.send(chunks[send_chunk_ind], dst_rank, group)
    # # After above, A holds complete a, B hold complete b, etc
    
    # #  ranks: A B C D 
    # #         a b c d
    # #         d a b c
    # #         c d a b
    # for i in range(ws-1):
    #     send_chunk_ind = (rank + ws - i) % ws
    #     dst_rank = (rank + 1) % ws
    #     write_chunk_ind = (rank + ws- 1 - i) % ws
    #     src_rank = (rank + ws - 1) % ws
    #     # print(f"Share {rank=}, {send_chunk_ind=}->{dst_rank=}, {src_rank=}->{write_chunk_ind=}.")
    #     if rank % 2 == 0:
    #         dist.send(chunks[send_chunk_ind], dst_rank, group)
    #         dist.recv(chunks[write_chunk_ind], src_rank, group)
    #     else:
    #         dist.recv(chunks[write_chunk_ind], src_rank, group)
    #         dist.send(chunks[send_chunk_ind], dst_rank, group)
    
    # # chunk is a view, so already in place
    # return tensor


    # """ v1
    # out = torch.zeros_like(tensor)
    # for i in range(ws):
    #     if rank == i:
    #         dist.send(tensor, dst=(i+1)%ws, group=group)
    #     if rank == i + 1:
    #         dist.recv(out, src=i, group=group)
    #         tensor = tensor + out

    # if rank == 0:
    #     dist.recv(out, src=ws-1, group=group)

    # for i in range(ws):
    #     if rank == i:
    #         dist.send(out, dst=(i+1)%ws, group=group)
    #     if rank == i + 1:
    #         dist.recv(out, src=i, group=group)

    # if rank == 0:
    #     dist.recv(out, src=ws-1, group=group)
    # """
    # return out
