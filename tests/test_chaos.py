"""Crash the runtime at every possible write and prove recovery is correct.

For a workflow that mixes plain steps, a retried step, a durable timer, a
signal and recorded time, the test first measures how many journal writes a
clean execution performs. It then re-runs the workflow once per write position
(and for every pair of positions), killing the "process" just before that
write, restarting on a brand-new runtime and driving the run to completion.

Invariants checked for every crash schedule:

* the final result equals the crash-free result;
* every external effect is committed exactly once per idempotency key;
* no effect is invoked more than once per crash that could have interrupted it;
* a step whose completion was journaled is never invoked again.
"""

from __future__ import annotations

import itertools

import pytest

from conftest import FakeClock, SimulatedCrash
from revenant import RetryPolicy, Runtime, workflow


class World:
    """The external systems the workflow talks to. Survives 'process' crashes."""

    def __init__(self) -> None:
        self.ledger: dict[str, int] = {}
        self.invocations: list[str] = []
        self.flaky_failed: set[str] = set()

    def charge(self, amount: int, idempotency_key: str) -> dict:
        self.invocations.append(idempotency_key)
        self.ledger.setdefault(idempotency_key, amount)
        return {"charged": self.ledger[idempotency_key]}

    def flaky(self, idempotency_key: str) -> str:
        self.invocations.append(idempotency_key)
        if idempotency_key not in self.flaky_failed:
            self.flaky_failed.add(idempotency_key)
            raise ConnectionError("transient")
        self.ledger.setdefault(idempotency_key, 1)
        return "sent"


WORLD: World


@workflow(name="chaos.workflow")
def chaos_workflow(ctx, inp):
    ctx.now()
    a = ctx.step("charge-a", WORLD.charge, inp["a"])
    ctx.sleep(60, "settle")
    mail = ctx.step("notify", WORLD.flaky, retry=RetryPolicy(max_attempts=3, initial_delay=0))
    decision = ctx.wait_signal("confirm")
    b = ctx.step("charge-b", WORLD.charge, inp["b"])
    return {"a": a, "mail": mail, "decision": decision, "b": b, "id_len": len(ctx.uuid())}


def drive(db: str, crash_before: set[int]) -> tuple[dict, World, int, int]:
    """Run to completion, crashing before each append whose ordinal is listed."""
    global WORLD
    WORLD = World()
    clock = FakeClock()
    appends = 0
    crashes = 0
    pending = set(crash_before)
    run_id = None

    def hook(event_type: str, seq: int) -> None:
        nonlocal appends
        appends += 1
        if appends in pending:
            pending.discard(appends)
            appends -= 1  # the write never happened
            raise SimulatedCrash

    for incarnation in range(100):
        # A new Runtime per incarnation models a fresh process after a crash.
        rt = Runtime(db, worker_id=f"w{incarnation}", clock=clock, sleeper=clock.sleep, heartbeat=False)
        rt.journal.before_append = hook
        try:
            if run_id is None:
                run_id = rt.submit(chaos_workflow, {"a": 10, "b": 20})
            result = rt.resume(run_id, force=True)
            if result.completed:
                events = rt.journal.events(run_id)
                assert [e.seq for e in events] == list(range(len(events)))
                return result.result, WORLD, appends, crashes
            assert result.suspended, result
            if result.wake_at is not None:
                clock.advance(61)
            else:
                rt.signal(run_id, result.waiting_on, {"ok": True})
        except SimulatedCrash:
            crashes += 1
        finally:
            rt.close()
    raise AssertionError("run did not converge")


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    result, world, appends, crashes = drive(str(tmp_path_factory.mktemp("base") / "j.db"), set())
    assert crashes == 0
    assert sorted(world.ledger.values()) == [1, 10, 20]
    # each charge once; the flaky step twice (one transient failure)
    assert len(world.invocations) == 4
    return result, appends


def check(result, world, crashes, baseline_result):
    assert result == baseline_result
    assert sorted(world.ledger.values()) == [1, 10, 20], "effects must commit exactly once"
    counts = {key: world.invocations.count(key) for key in world.ledger}
    extra = sum(counts.values()) - 3
    # at-least-once: each crash can cost at most one repeated invocation, and
    # the flaky step legitimately needs up to two.
    assert 0 <= extra <= crashes + 1, counts


def test_crash_before_every_single_write(tmp_path, baseline):
    baseline_result, total = baseline
    assert total >= 15
    for position in range(1, total + 1):
        result, world, _, crashes = drive(str(tmp_path / f"single-{position}.db"), {position})
        assert crashes == 1, position
        check(result, world, crashes, baseline_result)


def test_crash_before_every_pair_of_writes(tmp_path, baseline):
    baseline_result, total = baseline
    for first, second in itertools.combinations(range(1, total + 1), 2):
        result, world, _, crashes = drive(str(tmp_path / f"pair-{first}-{second}.db"), {first, second})
        # Usually both crashes fire. If the first one swallows the journaling of
        # the transient failure, the recovered run needs one write fewer and the
        # second position may never be reached.
        assert 1 <= crashes <= 2, (first, second)
        check(result, world, crashes, baseline_result)


def test_completed_steps_are_never_reinvoked(tmp_path, baseline):
    """Crash right after 'charge-a' is journaled: it must not run again."""
    baseline_result, _ = baseline
    db = str(tmp_path / "after-complete.db")
    # appends: 1 run_started, 2-3 now, 4 charge-a started, 5 charge-a completed, 6 timer_started
    result, world, _, _ = drive(db, {6})
    assert result == baseline_result
    key = next(k for k, v in world.ledger.items() if v == 10)
    assert world.invocations.count(key) == 1
