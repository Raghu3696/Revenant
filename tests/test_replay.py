from __future__ import annotations

import json

import pytest

from revenant import (
    NO_RETRY,
    NonDeterminismError,
    NonRetryableError,
    RetryPolicy,
    StepFailed,
    workflow,
)


def test_steps_run_once_and_replay_returns_recorded_values(rt, make_runtime):
    calls = []

    def fetch(n):
        calls.append(n)
        return {"n": n, "pair": (n, n + 1)}

    @workflow(name="replay.basic")
    def wf(ctx, inp):
        a = ctx.step("a", fetch, inp["x"])
        ctx.wait_signal("go")
        b = ctx.step("b", fetch, a["n"] + 1)
        return [a, b]

    first = rt.start(wf, {"x": 1})
    assert first.suspended and first.waiting_on == "go"
    assert calls == [1]

    rt.signal(first.run_id, "go")
    other = make_runtime(worker_id="worker-b")
    done = other.resume(first.run_id)
    assert done.completed
    # step "a" was replayed, not re-executed; tuples are normalised to lists
    # identically on the live and replayed paths.
    assert calls == [1, 2]
    assert done.result == [{"n": 1, "pair": [1, 2]}, {"n": 2, "pair": [2, 3]}]

    # resuming a finished run is a no-op that returns the stored result
    again = rt.resume(first.run_id)
    assert again.result == done.result and calls == [1, 2]


def test_recorded_time_random_and_uuid_are_stable_across_resume(rt, clock):
    @workflow(name="replay.sources")
    def wf(ctx, inp):
        seen = {"now": ctx.now(), "rand": ctx.random(), "id": ctx.uuid()}
        ctx.wait_signal("go")
        return seen

    first = rt.start(wf)
    events = rt.journal.events(first.run_id)
    recorded = {e.name: e.payload["result"] for e in events if e.type == "step_completed"}
    clock.advance(500)
    rt.signal(first.run_id, "go")
    done = rt.resume(first.run_id)
    assert done.result == {"now": recorded["now"], "rand": recorded["random"], "id": recorded["uuid"]}


def test_retries_then_succeeds_with_backoff(rt, clock):
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("boom")
        return "ok"

    @workflow(name="retry.flaky")
    def wf(ctx, inp):
        return ctx.step("flaky", flaky, retry=RetryPolicy(max_attempts=4, initial_delay=1.0, jitter=0.0))

    result = rt.start(wf)
    assert result.result == "ok" and len(attempts) == 3
    assert clock.slept == [1.0, 2.0]
    types = [e.type for e in rt.journal.events(result.run_id)]
    assert types.count("step_attempt_failed") == 2


def test_exhausted_step_failure_is_catchable_and_replayed_without_rerunning(rt):
    calls = []

    def broken():
        calls.append(1)
        raise ValueError("nope")

    @workflow(name="retry.exhausted")
    def wf(ctx, inp):
        try:
            ctx.step("broken", broken, retry=RetryPolicy(max_attempts=2, initial_delay=0))
        except StepFailed as exc:
            outcome = {"failed": exc.step, "attempts": exc.attempts, "type": exc.error["type"]}
        ctx.wait_signal("go")
        return outcome

    first = rt.start(wf)
    assert len(calls) == 2
    rt.signal(first.run_id, "go")
    done = rt.resume(first.run_id)
    assert done.result == {"failed": "broken", "attempts": 2, "type": "ValueError"}
    assert len(calls) == 2  # the failure was replayed from the journal


def test_non_retryable_error_skips_retries(rt):
    calls = []

    def fatal():
        calls.append(1)
        raise NonRetryableError("bad request")

    @workflow(name="retry.fatal")
    def wf(ctx, inp):
        return ctx.step("fatal", fatal, retry=RetryPolicy(max_attempts=5, initial_delay=0))

    result = rt.start(wf)
    assert result.status == "failed" and len(calls) == 1
    assert result.error["type"] == "StepFailed"


def test_unserialisable_result_fails_the_step(rt):
    @workflow(name="serialise.bad")
    def wf(ctx, inp):
        return ctx.step("bad", lambda: object(), retry=NO_RETRY)

    result = rt.start(wf)
    assert result.status == "failed"
    assert "JSON-serialisable" in result.error["message"]


def test_uncaught_exception_fails_the_run_with_traceback(rt):
    @workflow(name="fail.uncaught")
    def wf(ctx, inp):
        ctx.step("fine", lambda: 1)
        raise RuntimeError("workflow bug")

    result = rt.start(wf)
    assert result.status == "failed"
    assert result.error["type"] == "RuntimeError" and "workflow bug" in result.error["traceback"]
    assert rt.journal.events(result.run_id)[-1].type == "run_failed"


def test_changed_code_is_detected_and_run_is_stalled_not_failed(rt):
    def make(step_name):
        @workflow(name="nondeterminism.rename")
        def wf(ctx, inp):
            ctx.step(step_name, lambda: 1)
            ctx.wait_signal("go")
            return "done"

        return wf

    first = rt.start(make("original"))
    rt.signal(first.run_id, "go")
    make("renamed")  # re-register different code under the same workflow name
    with pytest.raises(NonDeterminismError, match="history mismatch"):
        rt.resume(first.run_id)
    assert rt.journal.get_run(first.run_id).status == "stalled"

    make("original")  # deploy the fix; the stalled run continues untouched
    assert rt.resume(first.run_id).result == "done"


def test_changed_arguments_are_detected(rt):
    def make(arg):
        @workflow(name="nondeterminism.args")
        def wf(ctx, inp):
            ctx.step("s", lambda x: x, arg)
            ctx.wait_signal("go")

        return wf

    first = rt.start(make(1))
    rt.signal(first.run_id, "go")
    make(2)
    with pytest.raises(NonDeterminismError, match="different arguments"):
        rt.resume(first.run_id)


def test_workflow_that_returns_early_on_replay_is_detected(rt):
    def make(short):
        @workflow(name="nondeterminism.short")
        def wf(ctx, inp):
            if short:
                return "early"
            ctx.step("s", lambda: 1)
            ctx.wait_signal("go")

        return wf

    first = rt.start(make(False))
    make(True)
    with pytest.raises(NonDeterminismError, match="unreplayed"):
        rt.resume(first.run_id)


def test_idempotency_key_is_stable_across_retries(rt):
    keys = []

    def charge(amount, idempotency_key):
        keys.append(idempotency_key)
        if len(keys) == 1:
            raise TimeoutError("unknown outcome")
        return amount

    @workflow(name="idempotency.key")
    def wf(ctx, inp):
        return ctx.step("charge", charge, 5, retry=RetryPolicy(initial_delay=0))

    result = rt.start(wf)
    assert result.result == 5
    assert len(keys) == 2 and keys[0] == keys[1] == f"{result.run_id}:1"


def test_live_and_replayed_results_serialise_identically(rt):
    """Key order must match too, or prompts built from results would drift."""
    seen = []

    @workflow(name="replay.key_order")
    def wf(ctx, inp):
        value = ctx.step("unordered", lambda: {"zeta": 1, "alpha": {"y": 2, "b": 3}})
        seen.append(json.dumps(value))
        ctx.wait_signal("go")
        return value

    first = rt.start(wf)
    rt.signal(first.run_id, "go")
    rt.resume(first.run_id)
    assert seen[0] == seen[1] == '{"alpha": {"b": 3, "y": 2}, "zeta": 1}'
