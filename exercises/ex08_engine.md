# ex08 — DeepSpeed-style engine

Expose the existing ZeRO stage implementations through `MiniDeepSpeedEngine`. Compare an engine-owned `backward()/step()` abstraction with PyTorch module/autograd integration.

## Integration hints

The engine owns orchestration, not a new sharding algorithm:

- `__call__` delegates to the original module for stages 1/2 and to the
  `MiniZeRO3` wrapper for stage 3. `engine.module` must remain the exact module
  supplied by the caller.
- `backward(loss)` may call ordinary `loss.backward()`; the stage-3 parameter
  hooks installed in ex07 participate automatically.
- Before a stage-1 optimizer step, arrange the mean gradient all-reduce that
  ex05 intentionally assumes. Stage 2 performs its own reduce-scatter. Stage 3
  leaves local gradient shards for an optimizer over the sharded parameters.
- `step()` should perform the stage-specific update, clear gradients/state for
  the next iteration, and leave/materialize parameters in the state required
  by the next forward.

The test deliberately gives each rank different input. Cross-rank agreement
alone is insufficient; every stage must match an ordinary optimizer driven by
the global mean gradient.
