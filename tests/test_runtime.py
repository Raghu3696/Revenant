from __future__ import annotations

import pytest

from revenant import (
    Budget,
    BudgetExceeded,
    LeaseHeld,
    LeaseLost,
    RevenantError,
    RunNotFound,
    ScriptedModel,
    WorkflowNotFound,
    workflow,
)


def test_durable_timer_suspends_and_worker_wakes_it(rt, clock):
    @workflow(name="timer.basic")
    def wf(ctx, inp):
        ctx.step("before", lambda: 1)
        ctx.sleep(3600, "cooldown")
        return ctx.step("after", lambda: 2)

    first = rt.start(wf)
    assert first.suspended and first.wake_at == clock.now + 3600
    assert rt.pending(first.run_id) == {"timer": True, "wake_at": first.wake_at}

    assert rt.run_worker(stop_when_idle=True) == 0  # not due yet
    clock.advance(3601)
    assert rt.run_worker(stop_when_idle=True) == 1
    done = rt.journal.get_run(first.run_id)
    assert done.status == "completed" and done.result == 2


def test_run_waits_out_timers_in_process(rt, clock):
    @workflow(name="timer.inline")
    def wf(ctx, inp):
        ctx.sleep(10)
        ctx.sleep(5, "second")
        return "woke"

    result = rt.run(wf)
    assert result.result == "woke" and clock.slept == [10, 5]


def test_resuming_before_a_timer_is_due_keeps_it_suspended(rt, clock):
    @workflow(name="timer.early")
    def wf(ctx, inp):
        ctx.sleep(100)
        return "done"

    first = rt.start(wf)
    clock.advance(50)
    again = rt.resume(first.run_id)
    assert again.suspended and again.wake_at == first.wake_at  # deadline did not move
    clock.advance(50)
    assert rt.resume(first.run_id).result == "done"


def test_signals_are_delivered_in_order_and_worker_picks_them_up(rt):
    @workflow(name="signal.order")
    def wf(ctx, inp):
        return [ctx.wait_signal("msg"), ctx.wait_signal("msg")]

    first = rt.start(wf)
    rt.signal(first.run_id, "other", "ignored")
    assert rt.run_worker(stop_when_idle=True) == 0  # unrelated signal does not wake it
    rt.signal(first.run_id, "msg", "one")
    rt.signal(first.run_id, "msg", "two")
    assert rt.run_worker(stop_when_idle=True) == 1
    assert rt.journal.get_run(first.run_id).result == ["one", "two"]


def test_signal_sent_before_the_wait_is_not_lost(rt):
    @workflow(name="signal.early")
    def wf(ctx, inp):
        ctx.step("work", lambda: 1)
        return ctx.wait_signal("ready")

    run_id = rt.submit(wf)
    rt.signal(run_id, "ready", {"ok": True})
    assert rt.resume(run_id).result == {"ok": True}


def test_signal_to_unknown_run_raises(rt):
    with pytest.raises(RunNotFound):
        rt.signal("missing", "x")


def test_submit_then_worker_runs_pending_runs(rt):
    @workflow(name="worker.pending")
    def wf(ctx, inp):
        return inp * 2

    ids = [rt.submit(wf, n) for n in (1, 2, 3)]
    assert rt.run_worker(stop_when_idle=True) == 3
    assert [rt.journal.get_run(i).result for i in ids] == [2, 4, 6]


def test_unknown_workflow(rt):
    with pytest.raises(WorkflowNotFound):
        rt.start("does.not:exist")
    run_id = rt.submit("never-registered")
    assert rt.run_worker(stop_when_idle=True) == 0
    assert rt.journal.get_run(run_id).status == "pending"


def test_workflow_resolves_by_import_path(rt):
    result = rt.start("json:dumps")  # any importable callable(ctx, input)
    assert result.status == "failed"  # json.dumps(ctx, None) is nonsense, but it resolved


def test_step_budget(rt):
    @workflow(name="budget.steps")
    def wf(ctx, inp):
        done = 0
        try:
            for i in range(10):
                ctx.step(f"s{i}", lambda: 1)
                done += 1
        except BudgetExceeded:
            pass
        return done

    assert rt.start(wf, budget=Budget(max_steps=3)).result == 3


def test_token_cost_and_call_budgets_are_enforced_and_replayed(rt):
    model = ScriptedModel(
        lambda messages, tools: {"content": "x", "usage": {"input_tokens": 600, "output_tokens": 400}},
        input_price=2.0,
        output_price=10.0,
    )

    @workflow(name="budget.llm")
    def wf(ctx, inp):
        calls = 0
        try:
            while True:
                ctx.llm(model, [{"role": "user", "content": f"q{calls}"}])
                calls += 1
        except BudgetExceeded as exc:
            reason = str(exc)
        ctx.wait_signal("go")
        return {"calls": calls, "reason": reason}

    def run(budget):
        first = rt.start(wf, budget=budget)
        rt.signal(first.run_id, "go")
        return rt.resume(first.run_id)  # the resume replays every call and re-derives usage

    by_tokens = run(Budget(max_tokens=2500))
    assert by_tokens.result["calls"] == 3 and "token budget" in by_tokens.result["reason"]
    assert by_tokens.usage["total_tokens"] == 3000
    # 600 in * $2/M + 400 out * $10/M = $0.0052 per call
    assert by_tokens.usage["cost_usd"] == pytest.approx(0.0156)

    by_cost = run(Budget(max_cost_usd=0.006))
    assert by_cost.result["calls"] == 2 and "cost budget" in by_cost.result["reason"]

    by_calls = run(Budget(max_llm_calls=1))
    assert by_calls.result["calls"] == 1 and "LLM call budget" in by_calls.result["reason"]


def test_uncaught_budget_error_fails_the_run(rt):
    @workflow(name="budget.uncaught")
    def wf(ctx, inp):
        ctx.step("a", lambda: 1)
        ctx.step("b", lambda: 1)

    result = rt.start(wf, budget=Budget(max_steps=1))
    assert result.status == "failed" and result.error["type"] == "BudgetExceeded"


def test_fork_retries_a_failed_run_with_fixed_code(rt):
    calls = []

    def make(buggy):
        @workflow(name="fork.fix")
        def wf(ctx, inp):
            a = ctx.step("expensive", lambda: calls.append("expensive") or 10)
            if buggy:
                raise RuntimeError("bug after the expensive step")
            return a + 1

        return wf

    failed = rt.start(make(True))
    assert failed.status == "failed"
    make(False)
    fork_id = rt.fork(failed.run_id)
    result = rt.resume(fork_id)
    assert result.result == 11
    assert calls == ["expensive"]  # the fork reused the journaled result
    fork = rt.journal.get_run(fork_id)
    assert fork.parent_run_id == failed.run_id and fork.forked_at_seq == 3
    assert rt.journal.get_run(failed.run_id).status == "failed"  # original untouched


def test_fork_at_earlier_point_reexecutes_the_tail(rt):
    counter = {"n": 0}

    def bump():
        counter["n"] += 1
        return counter["n"]

    @workflow(name="fork.branch")
    def wf(ctx, inp):
        return [ctx.step("one", bump), ctx.step("two", bump)]

    original = rt.start(wf)
    assert original.result == [1, 2]
    # events: 0 run_started, 1-2 step one, 3-4 step two, 5 run_completed
    fork_id = rt.fork(original.run_id, at_seq=3)
    assert rt.resume(fork_id).result == [1, 3]
    with pytest.raises(ValueError):
        rt.fork(original.run_id, at_seq=0)


def test_fork_can_raise_the_budget(rt):
    @workflow(name="fork.budget")
    def wf(ctx, inp):
        return [ctx.step(f"s{i}", lambda i=i: i) for i in range(3)]

    failed = rt.start(wf, budget=Budget(max_steps=2))
    assert failed.status == "failed"
    fork_id = rt.fork(failed.run_id, budget=Budget(max_steps=5))
    assert rt.resume(fork_id).result == [0, 1, 2]


def test_lease_blocks_second_worker_until_expiry_then_fences_the_first(make_runtime, clock):
    a = make_runtime(worker_id="a", lease_ttl=30)
    b = make_runtime(worker_id="b", lease_ttl=30)
    state = {}

    def slow(idempotency_key):
        state.setdefault("calls", []).append(idempotency_key)
        if "takeover" not in state:
            # While worker a is stuck in this step, its lease expires and
            # worker b takes the run over and finishes it.
            state["takeover"] = True
            with pytest.raises(LeaseHeld):
                b.resume(state["run_id"])
            clock.advance(31)
            state["b_result"] = b.resume(state["run_id"])
        return "done"

    @workflow(name="lease.fencing")
    def wf(ctx, inp):
        return ctx.step("slow", slow)

    state["run_id"] = a.submit(wf)
    with pytest.raises(LeaseLost):
        a.resume(state["run_id"])  # a wakes up late; its write is rejected

    assert state["b_result"].result == "done"
    assert len(state["calls"]) == 2 and len(set(state["calls"])) == 1  # same idempotency key
    events = a.journal.events(state["run_id"])
    assert [e.type for e in events] == ["run_started", "step_started", "step_completed", "run_completed"]


def test_force_resume_takes_over_immediately(make_runtime):
    a = make_runtime(worker_id="a")
    b = make_runtime(worker_id="b")

    @workflow(name="lease.force")
    def wf(ctx, inp):
        ctx.wait_signal("go")
        return "ok"

    run_id = a.submit(wf)
    assert a.journal.acquire_lease(run_id, "a", 30)  # a holds the run and "dies"
    with pytest.raises(LeaseHeld):
        b.resume(run_id)
    assert b.resume(run_id, force=True).suspended


def test_worker_recovers_run_orphaned_by_a_dead_worker(make_runtime, clock):
    from conftest import SimulatedCrash

    a = make_runtime(worker_id="a", lease_ttl=10)
    b = make_runtime(worker_id="b", lease_ttl=10)

    @workflow(name="lease.orphan")
    def wf(ctx, inp):
        ctx.step("one", lambda: 1)
        return ctx.step("two", lambda: 2)

    def die(event_type, seq):
        if seq == 3:
            raise SimulatedCrash

    a.journal.before_append = die
    run_id = a.submit(wf)
    with pytest.raises(SimulatedCrash):
        a.resume(run_id)
    assert b.run_worker(stop_when_idle=True) == 0  # lease still live
    clock.advance(11)
    assert b.run_worker(stop_when_idle=True) == 1
    assert b.journal.get_run(run_id).result == 2


def test_approve_requires_a_pending_approval(rt):
    @workflow(name="approve.none")
    def wf(ctx, inp):
        ctx.wait_signal("plain")

    first = rt.start(wf)
    with pytest.raises(RevenantError, match="not waiting for an approval"):
        rt.approve(first.run_id)
    assert rt.pending(first.run_id) == {"signal": "plain", "details": None}
