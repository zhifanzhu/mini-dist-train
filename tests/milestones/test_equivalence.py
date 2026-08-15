import torch
from mini_dist.milestones import run_final_equivalence


def test_all_training_realizations_match_reference_update():
    results = run_final_equivalence(world_size=2, steps=2)
    required = {"baseline", "ddp", "zero1", "zero2", "zero3", "fsdp1", "fsdp2"}
    assert set(results) == required
    reference = results["baseline"]
    for name, params in results.items():
        assert params.shape == reference.shape, name
        torch.testing.assert_close(params, reference, rtol=2e-5, atol=2e-6, msg=lambda m, n=name: f"{n}: {m}")
