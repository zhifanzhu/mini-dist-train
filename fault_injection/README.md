# Fault-injection labs

These programs are intentionally incorrect. Run them with a short process-group timeout and debug logging. Do not treat a timeout as a test-harness failure: the failure is the subject of the exercise.

For each script:

1. write the ordered collective sequence for every rank;
2. identify the first point where ranks diverge;
3. predict which ranks block/fail;
4. run it;
5. compare the observed exception/watchdog/debug output to your prediction.

GPU/NCCL versions are most instructive, but several cases can also be observed with Gloo.
