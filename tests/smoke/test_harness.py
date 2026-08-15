from pathlib import Path
import torch
import torch.distributed as dist
from tests._dist_test_utils import run_gloo


def _worker(rank, world_size):
    x = torch.tensor([float(rank + 1)])
    dist.all_reduce(x)
    assert x.item() == sum(range(1, world_size + 1))


def test_repository_layout_exists():
    root = Path(__file__).resolve().parents[2]
    for rel in ["README.md", "AGENTS.md", "course.yml", "mini_dist", "tests/exercises"]:
        assert (root / rel).exists(), rel


def test_gloo_spawn_harness():
    run_gloo(2, _worker)
