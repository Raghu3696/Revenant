"""The runtime: starts, resumes, signals, forks and schedules runs."""

from __future__ import annotations

import importlib
import logging
import os
import socket
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .context import Context, to_json
from .errors import (
    LeaseHeld,
    LeaseLost,
    NonDeterminismError,
    RevenantError,
    Suspend,
    WorkflowNotFound,
)
from .journal import TERMINAL_EVENTS, TERMINAL_STATUSES, Event, Journal, RunRecord
from .policy import Budget, RetryPolicy

log = logging.getLogger("revenant")

WorkflowFn = Callable[[Context, Any], Any]

_REGISTRY: dict[str, WorkflowFn] = {}


def workflow(fn: WorkflowFn | None = None, *, name: str | None = None) -> Any:
    """Register a workflow function.

    Without ``name`` the workflow is registered as ``"module:qualname"``, which
    lets other processes (the CLI, a worker) import it on demand.
    """

    def register(func: WorkflowFn) -> WorkflowFn:
        key = name or f"{func.__module__}:{func.__qualname__}"
        _REGISTRY[key] = func
        func.__revenant_name__ = key  # type: ignore[attr-defined]
        return func

    return register(fn) if fn is not None else register


def resolve_workflow(name: str) -> WorkflowFn:
    """Find a workflow by registered name, importing ``module:attr`` if needed."""
    if name in _REGISTRY:
        return _REGISTRY[name]
    module_name, _, attr = name.partition(":")
    if attr:
        try:
            target: Any = importlib.import_module(module_name)
            for part in attr.split("."):
                target = getattr(target, part)
        except (ImportError, AttributeError) as exc:
            raise WorkflowNotFound(f"cannot import workflow {name!r}: {exc}") from exc
        if callable(target):
            _REGISTRY.setdefault(name, target)
            return target  # type: ignore[no-any-return]
    raise WorkflowNotFound(
        f"no workflow registered as {name!r}; import the module that defines it first"
    )


@dataclass(frozen=True)
class RunResult:
    """Outcome of driving a run as far as it can currently go."""

    run_id: str
    status: str
    result: Any = None
    error: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    wake_at: float | None = None
    waiting_on: str | None = None

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    @property
    def suspended(self) -> bool:
        return self.status == "suspended"


class _Heartbeat:
    """Keeps a run's lease alive while a long step is executing."""

    def __init__(self, journal: Journal, run_id: str, owner: str, ttl: float) -> None:
        self._stop = threading.Event()
        self._args = (journal, run_id, owner, ttl)
        self._thread = threading.Thread(target=self._loop, name="revenant-heartbeat", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        journal, run_id, owner, ttl = self._args
        while not self._stop.wait(ttl / 3.0):
            try:
                journal.renew_lease(run_id, owner, ttl)
            except Exception:  # pragma: no cover - best effort, fencing is the guard
                log.debug("lease heartbeat failed for run %s", run_id, exc_info=True)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


class Runtime:
    """Entry point for running durable workflows against one journal."""

    def __init__(
        self,
        path: str = "revenant.db",
        *,
        worker_id: str | None = None,
        lease_ttl: float = 30.0,
        default_retry: RetryPolicy | None = None,
        durability: str = "normal",
        heartbeat: bool = True,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if lease_ttl <= 0:
            raise ValueError("lease_ttl must be positive")
        self.clock = clock
        self.sleeper = sleeper
        self.lease_ttl = float(lease_ttl)
        self.default_retry = default_retry or RetryPolicy()
        self.heartbeat = heartbeat
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.journal = Journal(path, clock=clock, durability=durability)

    def close(self) -> None:
        self.journal.close()

    def __enter__(self) -> Runtime:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --------------------------------------------------------------- starting

    def submit(
        self,
        wf: WorkflowFn | str,
        input: Any = None,
        *,
        run_id: str | None = None,
        budget: Budget | None = None,
    ) -> str:
        """Create a run without executing it; a worker will pick it up."""
        name = self._name_of(wf)
        run_id = run_id or uuid.uuid4().hex
        options = {"budget": budget.to_dict()} if budget else {}
        self.journal.create_run(run_id, name, to_json(input), options)
        return run_id

    def start(
        self,
        wf: WorkflowFn | str,
        input: Any = None,
        *,
        run_id: str | None = None,
        budget: Budget | None = None,
    ) -> RunResult:
        """Create a run and drive it until it completes, fails or suspends."""
        return self._execute(self.submit(wf, input, run_id=run_id, budget=budget))

    def run(
        self,
        wf: WorkflowFn | str,
        input: Any = None,
        *,
        run_id: str | None = None,
        budget: Budget | None = None,
    ) -> RunResult:
        """Like :meth:`start`, but also waits out durable timers in-process.

        Returns when the run finishes or is waiting for an external signal.
        """
        result = self.start(wf, input, run_id=run_id, budget=budget)
        while result.suspended and result.wake_at is not None:
            self.sleeper(max(0.0, result.wake_at - self.clock()))
            result = self._execute(result.run_id)
        return result

    def resume(self, run_id: str, *, force: bool = False) -> RunResult:
        """Continue a run from its journal.

        ``force=True`` takes the lease from its current owner. Use it when you
        know that worker is dead and do not want to wait for the lease to
        expire. It is safe even if you are wrong: the old worker is fenced off
        and its next write is rejected.
        """
        return self._execute(run_id, force=force)

    # ---------------------------------------------------------------- signals

    def signal(self, run_id: str, name: str, payload: Any = None) -> None:
        """Deliver a signal. It is consumed by the matching ``wait_signal``."""
        self.journal.add_signal(run_id, name, to_json(payload))

    def pending(self, run_id: str) -> dict[str, Any] | None:
        """What a suspended run is waiting for, or ``None`` if it is not waiting."""
        run = self.journal.get_run(run_id)
        if run.status != "suspended":
            return None
        if run.waiting_on is None:
            return {"timer": True, "wake_at": run.wake_at}
        for event in reversed(self.journal.events(run_id)):
            if event.type == "signal_waiting" and event.name == run.waiting_on:
                return {"signal": run.waiting_on, "details": event.payload.get("details")}
        return {"signal": run.waiting_on, "details": None}

    def approve(
        self, run_id: str, *, approved: bool = True, reason: str | None = None
    ) -> RunResult:
        """Answer the approval a run is waiting on, then resume it."""
        run = self.journal.get_run(run_id)
        if run.status != "suspended" or not (run.waiting_on or "").startswith("approve:"):
            raise RevenantError(f"run {run_id} is not waiting for an approval")
        assert run.waiting_on is not None
        self.signal(run_id, run.waiting_on, {"approved": approved, "reason": reason})
        return self.resume(run_id)

    # ------------------------------------------------------------------- fork

    def fork(
        self,
        run_id: str,
        at_seq: int | None = None,
        *,
        wf: WorkflowFn | str | None = None,
        budget: Budget | None = None,
    ) -> str:
        """Branch a new run off the first ``at_seq`` events of an existing one.

        The fork replays the shared prefix for free and executes everything
        after it afresh, optionally with different code (``wf``) or limits
        (``budget``). With ``at_seq=None`` it keeps the whole history except a
        terminal event, which is how a failed run is retried after a fix.
        """
        events = self.journal.events(run_id)
        keep = len(events)
        while keep > 0 and events[keep - 1].type in TERMINAL_EVENTS:
            keep -= 1
        if at_seq is not None:
            if at_seq < 1:
                raise ValueError("at_seq must be at least 1 (event 0 holds the run input)")
            keep = min(keep, at_seq)
        options = None
        if budget is not None:
            options = dict(self.journal.get_run(run_id).options)
            options["budget"] = budget.to_dict()
        new_id = uuid.uuid4().hex
        self.journal.fork(
            run_id,
            new_id,
            keep,
            workflow=self._name_of(wf) if wf is not None else None,
            options=options,
        )
        return new_id

    # ----------------------------------------------------------------- worker

    def run_worker(
        self,
        *,
        poll_interval: float = 0.5,
        stop_when_idle: bool = False,
        max_iterations: int | None = None,
    ) -> int:
        """Poll for due runs and drive them. Returns how many runs were driven.

        Picks up new runs, fired timers, delivered signals, and runs orphaned by
        a crashed worker once their lease has expired.
        """
        driven = 0
        iterations = 0
        while max_iterations is None or iterations < max_iterations:
            iterations += 1
            progressed = False
            for run_id in self.journal.due_runs(self.clock()):
                try:
                    self._execute(run_id)
                except LeaseHeld:
                    continue
                except (NonDeterminismError, WorkflowNotFound) as exc:
                    log.warning("skipping run %s: %s", run_id, exc)
                    continue
                driven += 1
                progressed = True
            if not progressed:
                if stop_when_idle:
                    break
                self.sleeper(poll_interval)
        return driven

    # -------------------------------------------------------------- internals

    @staticmethod
    def _name_of(wf: WorkflowFn | str) -> str:
        if isinstance(wf, str):
            return wf
        name = getattr(wf, "__revenant_name__", None)
        if name is None:
            workflow(wf)
            name = wf.__revenant_name__  # type: ignore[attr-defined]
        return str(name)

    @staticmethod
    def _result_of(run: RunRecord) -> RunResult:
        return RunResult(
            run_id=run.run_id,
            status=run.status,
            result=run.result,
            error=run.error,
            usage=run.usage,
            wake_at=run.wake_at,
            waiting_on=run.waiting_on,
        )

    def _execute(self, run_id: str, *, force: bool = False) -> RunResult:
        run = self.journal.get_run(run_id)
        if run.status in TERMINAL_STATUSES:
            return self._result_of(run)
        fn = resolve_workflow(run.workflow)
        if not self.journal.acquire_lease(run_id, self.worker_id, self.lease_ttl, force=force):
            raise LeaseHeld(
                f"run {run_id} is owned by worker {run.lease_owner!r}; wait for its lease to "
                "expire or resume with force=True"
            )
        heartbeat = (
            _Heartbeat(self.journal, run_id, self.worker_id, self.lease_ttl)
            if self.heartbeat
            else None
        )
        try:
            return self._drive(run, fn)
        finally:
            if heartbeat is not None:
                heartbeat.stop()

    def _drive(self, run: RunRecord, fn: WorkflowFn) -> RunResult:
        run_id = run.run_id
        journal = self.journal
        history: list[Event] = journal.events(run_id)
        ctx = Context(self, run, history)
        journal.update_run(run_id, status="running", wake_at=None, waiting_on=None, error=None)
        try:
            if history:
                if history[0].type != "run_started":
                    raise NonDeterminismError(f"run {run_id} has a corrupt history")
                ctx._pos = 1
            else:
                ctx._record("run_started", run.workflow, {"input": run.input})
            output = to_json(fn(ctx, run.input))
            if ctx.replaying:
                pending = history[ctx._pos]
                raise NonDeterminismError(
                    f"workflow returned while {len(history) - ctx._pos} recorded event(s) remain "
                    f"unreplayed (next: {pending.type} {pending.name!r})"
                )
            ctx._record("run_completed", run.workflow, {"result": output})
            journal.update_run(
                run_id, status="completed", result=output, usage=ctx.usage.to_dict()
            )
        except Suspend as pause:
            journal.update_run(
                run_id,
                status="suspended",
                wake_at=pause.wake_at,
                waiting_on=pause.waiting_on,
                usage=ctx.usage.to_dict(),
            )
        except LeaseLost:
            # Another worker owns the run now; its state is no longer ours to touch.
            raise
        except NonDeterminismError as exc:
            self._stall(run_id, exc)
            raise
        except Exception as exc:
            if ctx.replaying:
                # The code blew up before reaching the end of recorded history,
                # which the original execution did not do: the code has changed.
                error = NonDeterminismError(
                    f"workflow raised {type(exc).__name__}: {exc} while replaying event "
                    f"{ctx._pos} of {len(history)}"
                )
                self._stall(run_id, error)
                raise error from exc
            failure = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc)),
            }
            ctx._record("run_failed", run.workflow, {"error": failure})
            journal.update_run(
                run_id, status="failed", error=failure, usage=ctx.usage.to_dict()
            )
        journal.release_lease(run_id, self.worker_id)
        return self._result_of(journal.get_run(run_id))

    def _stall(self, run_id: str, exc: Exception) -> None:
        self.journal.update_run(
            run_id,
            status="stalled",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        self.journal.release_lease(run_id, self.worker_id)
