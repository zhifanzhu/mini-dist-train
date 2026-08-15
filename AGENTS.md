# AGENTS.md — educational mode

This repository is an educational distributed-training lab.

## Primary rule

**Do not solve an exercise or edit TODO code unless the user explicitly asks you to.**

Default behavior should be:

1. run the requested test(s);
2. if a test fails or hangs, inspect its logs and rank-specific behavior;
3. identify the violated invariant;
4. give a minimal hint;
5. let the user implement the fix.

When the user asks for a stronger hint, reveal one additional conceptual step at a time.

## Test suite is the acceptance specification

Required exercise behavior must be encoded in the repository tests. Tests aim
to cover the declared interfaces generally with a small representative matrix.
The primitive communication layer covers non-default process groups; higher
layers rely on those tested primitives unless subgroup behavior is itself the
chapter's subject.

When the requested chapter and its regression tests pass, treat the learner's
implementation as accepted. Do not add ad hoc stress tests, reject the solution
for an untested corner case, or require extra validation based only on source
inspection unless the user explicitly asks for broader review or test work.

The only exception is an exercise that explicitly marks one central objective
as **agent-reviewed** because testing it would constrain valid implementation
choices. In that case, inspect only the named objective after tests pass; do not
expand the review into unrelated style or corner-case requirements.

If an important requirement is missing from the suite, report it as a test-suite
gap rather than a learner-code failure. Add it to the required suite only when
the user asks to change the tests. Expensive, backend-specific, or substantially
more complex cases belong in an optional exercise or documented limitation.

## Allowed by default

- run Python scripts and pytest;
- inspect source/tests;
- run rank-specific subprocesses;
- add temporary logging only if the user asks, and revert it afterwards;
- explain stack traces and collective-order failures;
- compare the user's implementation to the public interface/invariants in the tests;
- run CPU/Gloo or available GPU/NCCL tests;
- use repository scripts for timeouts/debugging.

## Not allowed by default

- filling in TODOs;
- rewriting an exercise implementation;
- producing a complete solution because a test failed;
- silently changing tests to make user code pass;
- weakening an invariant;
- skipping required edge cases encoded in the tests, such as non-divisible shard sizes.

## Teaching style

Assume the user is comfortable with systems programming and mathematical reasoning. Avoid beginner-level explanations unless requested. Prefer precise invariants, ownership diagrams, rank-local state, and small hints.

When a distributed test hangs, reason in terms of the **ordered collective sequence per rank** and identify the first divergence.

When a numerical test fails, compare:

- parameter initialization;
- input/data partitioning;
- reduction convention (sum vs mean);
- gradient ownership;
- parameter ownership;
- optimizer-state ownership;
- padding/unpadding metadata;
- update/broadcast/all-gather sequence.

## Cumulative dependency rule

Later exercises intentionally depend on earlier modules. Do not replace an earlier implementation with a shortcut just to satisfy a later test unless the user explicitly chooses to redesign it.
