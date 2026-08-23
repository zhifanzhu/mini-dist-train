import copy
import torch
import torch.nn as nn
from mini_dist.engine import MiniDeepSpeedEngine
from tests._dist_test_utils import run_gloo


def _worker(rank, world_size, zero_stage):
    torch.manual_seed(1)
    module = nn.Linear(4, 1)
    reference = copy.deepcopy(module)
    engine = MiniDeepSpeedEngine(module, torch.optim.SGD, zero_stage=zero_stage, lr=0.1)
    x = torch.tensor([[float(rank), 1.0, 2.0, -1.0]])

    reference(x).sum().backward()
    for p in reference.parameters():
        torch.distributed.all_reduce(p.grad)
        p.grad.div_(world_size)
    torch.optim.SGD(reference.parameters(), lr=0.1).step()

    loss = engine(x).sum()
    engine.backward(loss)
    engine.step()
    assert engine.module is module
    for got, expected in zip(engine.module.parameters(), reference.parameters()):
        torch.testing.assert_close(got, expected)


def _run_stage1(rank, world_size):
    _worker(rank, world_size, 1)

def _run_stage2(rank, world_size):
    _worker(rank, world_size, 2)

def _run_stage3(rank, world_size):
    _worker(rank, world_size, 3)


def _stage3_two_step_worker(rank, world_size):
    torch.manual_seed(11)
    module = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1))
    reference = copy.deepcopy(module)
    engine = MiniDeepSpeedEngine(module, torch.optim.SGD, zero_stage=3, lr=0.05)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)

    for step in range(2):
        x = torch.tensor([[float(rank + step), 1.0, -2.0]])
        expected_output = reference(x)
        output = engine(x)

        if step > 0:
            locally_matches = torch.allclose(output, expected_output, rtol=2e-5, atol=2e-6)
            matches_on_all_ranks = torch.tensor(int(locally_matches))
            torch.distributed.all_reduce(
                matches_on_all_ranks, op=torch.distributed.ReduceOp.MIN
            )
            assert matches_on_all_ranks.item(), (
                f"Stage-3 forward at step {step} did not gather the parameter shards "
                "updated by the previous optimizer step"
            )

        expected_output.sum().backward()
        for parameter in reference.parameters():
            torch.distributed.all_reduce(parameter.grad)
            parameter.grad.div_(world_size)
        reference_optimizer.step()
        reference_optimizer.zero_grad(set_to_none=True)

        output.sum().backward()
        engine.step()
        for parameter in engine.module.parameters():
            parameter.grad = None

    for got, expected in zip(engine.module.parameters(), reference.parameters()):
        torch.testing.assert_close(
            got,
            expected,
            rtol=2e-5,
            atol=2e-6,
            msg="Stage-3 optimizer updates must remain authoritative across iterations",
        )


def test_engine_stage1_workflow():
    run_gloo(2, _run_stage1)


def test_engine_stage2_workflow():
    run_gloo(2, _run_stage2)


def test_engine_stage3_workflow():
    run_gloo(2, _run_stage3)


def test_engine_stage3_next_forward_uses_updated_parameter_shards():
    run_gloo(2, _stage3_two_step_worker)
