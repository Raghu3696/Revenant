# Design notes

## The problem

An agent run is a long sequence of expensive, non-deterministic, side-effecting calls. Three things go wrong in production:

1. **The process dies.** Restarting from zero repeats cost and repeats side effects.
2. **A human has to decide something.** Keeping a process alive for hours while someone reads an approval request is wasteful and fragile.
3. **Something went wrong and nobody can say what.** Without a faithful record there is nothing to inspect, replay or branch from.

All three are solved by one mechanism: an append-only journal of everything the run observed, plus the discipline that the run only observes the world through that journal.

## Event sourcing with replay

A run is its ordered event list. Status, usage and wake-up times in the `runs` table are caches for listing and scheduling; the events are authoritative.

| Event | Written when |
|---|---|
| `run_started` | First execution; holds the input |
| `step_started` | Before a step, model call or tool call runs; holds kind and fingerprint |
| `step_attempt_failed` | An attempt failed and will be retried |
| `step_completed` / `step_failed` | The step's final outcome |
| `timer_started` / `timer_fired` | A durable sleep began / elapsed |
| `signal_waiting` / `signal_received` | The run began waiting / a signal was consumed |
| `run_completed` / `run_failed` | Terminal outcome |

Resuming means calling the workflow function again with a cursor over the history. Each context operation consumes the events it would have written. If they are present it returns the recorded outcome; if the history ends it switches to live execution. A history that ends between `step_started` and its outcome identifies a step that was in flight during a crash.

Why re-execute the function instead of snapshotting the interpreter? Snapshots of Python frames are not portable across processes, versions or machines. A journal of observations is plain JSON, inspectable, and works with any code that follows one rule: go through `ctx` for anything non-deterministic.

### Normalising results

Live results are passed through the same JSON round-trip (with sorted keys) that stored results undergo. Without this, a step returning `{"b": 1, "a": 2}` would hand the workflow a dict in insertion order when live and in sorted order on replay. A workflow that serialises that dict into a prompt would then produce a different prompt on replay, and the next model call's fingerprint would not match. The test `test_live_and_replayed_results_serialise_identically` pins this.

## At-least-once, and why not more

Exactly-once execution of an arbitrary side effect is impossible: the process can die after the remote system applied the effect and before the local journal recorded it. Revenant therefore guarantees:

- a recorded outcome is never recomputed, and
- an in-flight step is re-run under a stable idempotency key, `"{run_id}:{seq of step_started}"`.

Systems that accept idempotency keys turn this into effectively-exactly-once. For systems that do not, the journal at least tells you precisely which step is ambiguous.

## Leases and fencing

A worker takes a lease on a run before driving it and extends it on every append; a heartbeat thread covers long steps. If the lease lapses the run is considered orphaned and any worker may take it.

Timeouts cannot distinguish a dead worker from a slow one, so mutual exclusion is not enforced by the lease alone. Every append runs in one `BEGIN IMMEDIATE` transaction that (1) checks the writer is still the lease owner and (2) inserts at `seq == len(history)` under a unique `(run_id, seq)` key. A worker that lost its lease gets `LeaseLost` on its next write and stops. `LeaseLost` derives from `BaseException` so workflow code with a broad `except Exception` cannot swallow it and keep going.

## Signals

External parties never write to a run's event list, which keeps the single-writer invariant simple. Signals go to a separate table. When a workflow reaches `wait_signal`, the owning worker moves the oldest matching signal into the journal as `signal_received` and marks it consumed, atomically. A signal that arrives before the wait is not lost; one that arrives while the run is suspended makes the run "due" for the worker loop.

## Budgets on replay

Usage is not stored as a counter that is incremented. It is re-derived on every execution from the journaled model responses, each of which carries the token counts and the cost computed at the price in effect when the call was made. Budget checks therefore trip at the same operation on replay as they did live, and changing a price table later does not rewrite history.

## Non-determinism

Each operation carries a fingerprint of its kind, name and JSON arguments. A model call's fingerprint covers the full message list and the tool names, but not the model name, so a fork may switch models for the part of the run that has not happened yet.

On mismatch the run becomes `stalled`: not failed, not retried by workers, and resumable as soon as compatible code is deployed. If the new behaviour is intended, fork from before the divergence.

## Forking

`fork(run, at_seq)` copies the first `at_seq` events into a new run. Because replay is positional, the fork fast-forwards through the shared prefix without executing anything, then goes live. Cutting between a `step_started` and its outcome re-executes that step; cutting right after `signal_waiting` re-asks the human question. With no `at_seq`, the fork drops only the terminal event, which is how a failed run is retried after a fix or with a larger budget.

## Possible next steps

- Parallel branches inside a workflow (`ctx.gather`), journaled per branch.
- Signal waits with timeouts.
- A Postgres journal for multi-machine deployments.
- Streaming model responses with journaled final output.
