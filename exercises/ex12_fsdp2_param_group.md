# ex12 — FSDP2-style parameter groups

Group multiple `FSDPParam` objects for communication. Pack shards into a temporary collective buffer, all-gather once, then restore individual parameter views. Backward performs grouped reduce-scatter.
