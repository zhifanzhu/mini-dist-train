import torch
import torch.nn as nn
from mini_dist.engine import MiniDeepSpeedEngine
from tests._dist_test_utils import run_gloo


def _worker(rank, world_size, zero_stage):
    torch.manual_seed(1)
    module = nn.Linear(4, 1)
    engine = MiniDeepSpeedEngine(module, torch.optim.SGD, zero_stage=zero_stage, lr=0.1)
    x = torch.tensor([[float(rank), 1.0, 2.0, -1.0]])
    loss = engine(x).sum()
    engine.backward(loss)
    engine.step()
    assert engine.module is module


def _run_stage1(rank, world_size):
    _worker(rank, world_size, 1)

def _run_stage2(rank, world_size):
    _worker(rank, world_size, 2)

def _run_stage3(rank, world_size):
    _worker(rank, world_size, 3)


def test_engine_stage1_workflow():
    run_gloo(2, _run_stage1)


def test_engine_stage2_workflow():
    run_gloo(2, _run_stage2)


def test_engine_stage3_workflow():
    run_gloo(2, _run_stage3)
