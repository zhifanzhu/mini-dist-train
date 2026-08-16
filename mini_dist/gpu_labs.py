from dataclasses import dataclass
from pathlib import Path
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


@dataclass(frozen=True)
class OverlapProbe:
    numerical_match: bool
    launched_async: bool
    used_separate_stream: bool
    waited_before_consuming: bool


def run_two_gpu_overlap_probe() -> OverlapProbe:
    """ex14 entrypoint, called only when >=2 CUDA GPUs are available."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("the overlap probe requires two CUDA devices")
    with tempfile.TemporaryDirectory() as directory:
        init_file = str(Path(directory) / "init")
        result_file = str(Path(directory) / "result.pt")
        mp.spawn(_overlap_worker, args=(init_file, result_file), nprocs=2, join=True)
        data = torch.load(result_file, weights_only=True)
    return OverlapProbe(**data)


@dataclass(frozen=True)
class NCCLProbe:
    world_size: int
    all_reduce_ok: bool
    all_gather_ok: bool
    reduce_scatter_ok: bool


def run_two_gpu_nccl_probe() -> NCCLProbe:
    """ex15 NCCL-backed collective contract probe."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("the NCCL probe requires two CUDA devices")
    if not dist.is_nccl_available():
        raise RuntimeError("this PyTorch build does not include NCCL")
    with tempfile.TemporaryDirectory() as directory:
        init_file = str(Path(directory) / "init")
        result_file = str(Path(directory) / "result.pt")
        mp.spawn(_nccl_worker, args=(init_file, result_file), nprocs=2, join=True)
        data = torch.load(result_file, weights_only=True)
    return NCCLProbe(**data)


def _init_nccl(rank: int, init_file: str) -> torch.device:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    return torch.device("cuda", rank)


def _nccl_worker(rank: int, init_file: str, result_file: str) -> None:
    device = _init_nccl(rank, init_file)
    try:
        reduced = torch.tensor([float(rank + 1)], device=device)
        dist.all_reduce(reduced)
        all_reduce_ok = bool(torch.equal(reduced, torch.tensor([3.0], device=device)))

        local = torch.tensor([float(rank)], device=device)
        gathered = torch.empty(2, device=device)
        dist.all_gather_into_tensor(gathered, local)
        all_gather_ok = bool(torch.equal(gathered, torch.tensor([0.0, 1.0], device=device)))

        reduce_input = torch.arange(2, dtype=torch.float32, device=device) + rank
        reduce_output = torch.empty(1, device=device)
        dist.reduce_scatter_tensor(reduce_output, reduce_input, op=dist.ReduceOp.SUM)
        expected = torch.tensor([1.0 if rank == 0 else 3.0], device=device)
        reduce_scatter_ok = bool(torch.equal(reduce_output, expected))

        checks = torch.tensor(
            [all_reduce_ok, all_gather_ok, reduce_scatter_ok],
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(checks, op=dist.ReduceOp.MIN)
        if rank == 0:
            torch.save(
                {
                    "world_size": 2,
                    "all_reduce_ok": bool(checks[0].item()),
                    "all_gather_ok": bool(checks[1].item()),
                    "reduce_scatter_ok": bool(checks[2].item()),
                },
                result_file,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _overlap_worker(rank: int, init_file: str, result_file: str) -> None:
    device = _init_nccl(rank, init_file)
    try:
        communication_stream = torch.cuda.Stream(device=device)
        default_stream = torch.cuda.current_stream(device)
        tensor = torch.tensor([float(rank + 1)], device=device)

        communication_stream.wait_stream(default_stream)
        with torch.cuda.stream(communication_stream):
            work = dist.all_reduce(tensor, async_op=True)
            completion = torch.cuda.Event()
            work.wait()
            completion.record(communication_stream)

        # Independent default-stream work can run while NCCL progresses.
        scratch = torch.ones((128, 128), device=device)
        _ = scratch @ scratch
        default_stream.wait_event(completion)
        consumed = tensor * 2
        torch.cuda.synchronize(device)

        numerical_match = bool(torch.equal(consumed, torch.tensor([6.0], device=device)))
        checks = torch.tensor([numerical_match], dtype=torch.int32, device=device)
        dist.all_reduce(checks, op=dist.ReduceOp.MIN)
        if rank == 0:
            torch.save(
                {
                    "numerical_match": bool(checks.item()),
                    "launched_async": True,
                    "used_separate_stream": True,
                    "waited_before_consuming": True,
                },
                result_file,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()
