# Exercise map

Each chapter extends the same `mini_dist` package. Do not copy your previous implementation into a new folder; later chapters import it directly.

| ID | Topic | Main dependency | What changes |
|---|---|---|---|
| ex00 | process groups | — | rank/world-size/process-group state |
| ex01 | collectives | ex00 | sum/mean, gather, reduce-scatter contracts |
| ex02 | ring all-reduce | ex01 | build one collective from send/recv |
| ex02b | tree all-reduce | ex01 | build a balanced-tree collective from P2P |
| ex03 | MiniDDP | ex01 | replicated params + synchronized grads |
| ex04 | buckets/hooks | ex03 | readiness, packing, async launches |
| ex05 | ZeRO-1 | ex03 | shard optimizer-state ownership |
| ex06 | ZeRO-2 | ex05 | additionally shard reduced gradients |
| ex07 | ZeRO-3 | ex06 | additionally shard parameters |
| ex08 | DeepSpeed-style engine | ex05–07 | engine owns backward/step workflow |
| ex09 | FSDP1 flat param | ex07 | permanent flat representation + views |
| ex10 | FSDP1 hooks | ex09 | automatic unshard/reshard lifecycle |
| ex11 | FSDP2 param | ex07 | per-parameter sharded identity |
| ex12 | FSDP2 param group | ex11 | temp packing + grouped collectives |
| ex13 | fully_shard API | ex12 | bottom-up ownership + hooks |
| ex14 | overlap/streams | ex04/ex12 | async comm + CUDA streams/prefetch |
| ex15 | NCCL/failures | ex01/ex13 | backend contracts and failure propagation |
| ex16 | final equivalence | all core | compare numerics/memory/communication |

Every chapter has a short markdown brief in this directory.

The learner is assumed to know ordinary PyTorch tensor/module/optimizer use.
Less-common framework mechanisms that are necessary to enter an exercise—such
as autograd hooks, temporary parameter views, process-group rank translation,
or CUDA stream dependencies—are named in the matching brief. Discovering that
an unfamiliar mechanism exists is not intended to be the exercise; reasoning
through its distributed invariant and implementing it correctly is.

## Extended tensor-parallel track

These optional chapters branch from the collective layer and later recombine
with FSDP. They use raw process groups to expose the sharding/layout algebra
that PyTorch DTensor and JAX sharding APIs normally automate.

| ID | Topic | Main dependency | What changes |
|---|---|---|---|
| ex17 | TP linear primitives | ex01 | autograd-aware layout transitions + column/row linear |
| ex18 | TP MLP | ex17 | keep the intermediate feature activation sharded |
| ex19 | 2D TP + FSDP | ex13/ex18 | named DP/TP groups + shard TP-local parameters |
