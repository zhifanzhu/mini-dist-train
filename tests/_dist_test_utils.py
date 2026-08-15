from __future__ import annotations

import os
from pathlib import Path
import tempfile
import traceback
import torch.distributed as dist
import torch.multiprocessing as mp


def _wrapped_worker(rank: int, world_size: int, init_file: str, fn, args):
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        fn(rank, world_size, *args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_gloo(world_size: int, fn, *args):
    """Spawn a small local Gloo job.

    Worker functions must be top-level functions so `spawn` can pickle them.
    """
    with tempfile.TemporaryDirectory() as td:
        init_file = str(Path(td) / "pg_init")
        mp.spawn(
            _wrapped_worker,
            args=(world_size, init_file, fn, args),
            nprocs=world_size,
            join=True,
        )
