from mini_dist.distributed import DistributedContext
from tests._dist_test_utils import run_gloo


def _worker(rank, world_size):
    ctx = DistributedContext.from_default_group()
    assert ctx.rank == rank
    assert ctx.world_size == world_size


def test_context_reads_default_process_group():
    run_gloo(2, _worker)
