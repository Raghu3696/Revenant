"""The workflow context: every side effect goes through here.

A workflow is ordinary Python. What makes it durable is that it never touches
the outside world directly; it asks the context to. The context either performs
the operation and journals the outcome, or, when the journal already holds an
outcome for this position, returns that outcome without doing anything.

Re-running the function from the top therefore fast-forwards through
everything that already happened and continues live from the first operation
that has no recorded result.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import random
import uuid
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from .errors import BudgetExceeded, NonDeterminismError, StepFailed, Suspend
from .journal import Event, RunRecord
from .policy import Budget, RetryPolicy, Usage

if TYPE_CHECKING:
    from .agent import Tool
    from .models import Model
    from .runtime import Runtime


def to_json(value: Any) -> Any:
    """Round-trip ``value`` through JSON.

    Live results are normalised exactly the way replayed results are, so a
    workflow sees the same value (lists, not tuples; string keys; the same key
    order) whether a step just ran or is being replayed. Key order matters: a
    workflow that serialises a result into a prompt must produce the identical
    string on replay.
    """
    return json.loads(json.dumps(value, sort_keys=True))


def fingerprint(kind: str, name: str, data: Any) -> str:
    """Stable identity of an operation, used to detect non-determinism."""
    try:
        body = json.dumps([kind, name, data], sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        # Arguments that are not JSON-serialisable cannot be compared reliably;
        # fall back to identifying the step by kind and name only.
        body = json.dumps([kind, name])
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def _accepts_idempotency_key(fn: Callable[..., Any]) -> bool:
    try:
        return "idempotency_key" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


class Context:
    """Handle passed as the first argument to every workflow function."""

    def __init__(self, runtime: Runtime, run: RunRecord, history: list[Event]) -> None:
        self._runtime = runtime
        self._journal = runtime.journal
        self._history = history
        self._pos = 0
        self.run_id = run.run_id
        self.budget = Budget.from_dict(run.options.get("budget"))
        self.usage = Usage()

    # ------------------------------------------------------------- plumbing

    @property
    def replaying(self) -> bool:
        """True while recorded history is still being fast-forwarded."""
        return self._pos < len(self._history)

    def _peek(self) -> Event | None:
        return self._history[self._pos] if self._pos < len(self._history) else None

    def _record(
        self,
        type: str,
        name: str = "",
        payload: dict[str, Any] | None = None,
        *,
        consume_signal: int | None = None,
    ) -> Event:
        if self.replaying:
            pending = self._history[self._pos]
            raise NonDeterminismError(
                f"workflow tried to record {type} {name!r} at position {self._pos}, but the "
                f"journal already holds {pending.type} {pending.name!r} there"
            )
        event = self._journal.append(
            self.run_id,
            len(self._history),
            type,
            name,
            payload or {},
            owner=self._runtime.worker_id,
            lease_ttl=self._runtime.lease_ttl,
            consume_signal=consume_signal,
        )
        self._history.append(event)
        self._pos += 1
        return event

    def _replay(self, type: str, name: str, fp: str | None = None) -> Event | None:
        """Consume the next recorded event if it matches; ``None`` when live."""
        event = self._peek()
        if event is None:
            return None
        recorded_fp = event.payload.get("fingerprint")
        if (
            event.type != type
            or event.name != name
            or (fp is not None and recorded_fp is not None and recorded_fp != fp)
        ):
            detail = "" if event.type != type or event.name != name else " with different arguments"
            raise NonDeterminismError(
                f"history mismatch at event {event.seq}: journal has {event.type} "
                f"{event.name!r}, but the workflow now requests {type} {name!r}{detail}. "
                "Workflow code must issue the same operations in the same order on replay."
            )
        self._pos += 1
        return event

    # ----------------------------------------------------------------- steps

    def step(
        self,
        name: str,
        fn: Callable[..., Any],
        *args: Any,
        retry: RetryPolicy | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run ``fn(*args, **kwargs)`` durably and return its JSON-safe result.

        The result is journaled; on replay it is returned without calling
        ``fn``. If ``fn`` declares an ``idempotency_key`` parameter it receives
        a key that is stable across retries and crash recovery, to pass on to
        external systems.
        """
        return self._run_step("step", name, fn, args, kwargs, retry, fp_data=[list(args), kwargs])

    def _run_step(
        self,
        kind: str,
        name: str,
        fn: Callable[..., Any],
        args: Sequence[Any],
        kwargs: dict[str, Any],
        retry: RetryPolicy | None,
        *,
        fp_data: Any,
        meta: dict[str, Any] | None = None,
        counted: bool = True,
    ) -> Any:
        if counted:
            limit = self.budget.max_steps
            if limit is not None and self.usage.steps >= limit:
                raise BudgetExceeded(f"step budget of {limit} exhausted before {name!r}")
            self.usage.steps += 1
        policy = retry or self._runtime.default_retry
        fp = fingerprint(kind, name, fp_data)

        failures = 0
        started = self._replay("step_started", name, fp)
        if started is None:
            payload: dict[str, Any] = {"kind": kind, "fingerprint": fp}
            if meta:
                payload["meta"] = meta
            started = self._record("step_started", name, payload)
        else:
            while (event := self._peek()) is not None:
                if event.name != name or event.type not in (
                    "step_attempt_failed",
                    "step_completed",
                    "step_failed",
                ):
                    raise NonDeterminismError(
                        f"history mismatch at event {event.seq}: step {name!r} is followed by "
                        f"{event.type} {event.name!r}"
                    )
                self._pos += 1
                if event.type == "step_attempt_failed":
                    failures += 1
                elif event.type == "step_completed":
                    return event.payload["result"]
                else:
                    raise StepFailed(name, event.payload["error"], event.payload["attempts"])
            # History ends inside this step: it was in flight when the previous
            # worker died. Run it again (at-least-once) under the same key.

        key = f"{self.run_id}:{started.seq}"
        call_kwargs = dict(kwargs)
        if _accepts_idempotency_key(fn):
            call_kwargs["idempotency_key"] = key

        attempt = failures
        while True:
            attempt += 1
            try:
                raw = fn(*args, **call_kwargs)
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
                if policy.should_retry(exc, attempt):
                    self._record("step_attempt_failed", name, {"attempt": attempt, "error": error})
                    self._runtime.sleeper(policy.delay(attempt, key))
                    continue
                self._record("step_failed", name, {"attempts": attempt, "error": error})
                raise StepFailed(name, error, attempt) from exc
            try:
                result = to_json(raw)
            except (TypeError, ValueError) as exc:
                error = {
                    "type": "SerializationError",
                    "message": f"step result is not JSON-serialisable: {exc}",
                }
                self._record("step_failed", name, {"attempts": attempt, "error": error})
                raise StepFailed(name, error, attempt) from exc
            self._record("step_completed", name, {"result": result})
            return result

    # ------------------------------------------------- deterministic helpers

    def now(self) -> float:
        """Current UNIX time, recorded once and replayed thereafter."""
        return float(
            self._run_step("now", "now", self._runtime.clock, (), {}, None, fp_data=None,
                           counted=False)
        )

    def random(self) -> float:
        """A random float in [0, 1), recorded once and replayed thereafter."""
        return float(
            self._run_step("random", "random", random.random, (), {}, None, fp_data=None,
                           counted=False)
        )

    def uuid(self) -> str:
        """A random UUID, recorded once and replayed thereafter."""
        return str(
            self._run_step("uuid", "uuid", lambda: str(uuid.uuid4()), (), {}, None, fp_data=None,
                           counted=False)
        )

    # ---------------------------------------------------------------- timers

    def sleep(self, seconds: float, name: str = "sleep") -> None:
        """Durable timer. The process may exit; the run resumes when it is due."""
        event = self._replay("timer_started", name)
        if event is None:
            wake_at = self._runtime.clock() + max(0.0, float(seconds))
            event = self._record("timer_started", name, {"wake_at": wake_at})
        fired = self._peek()
        if fired is not None:
            if fired.type != "timer_fired" or fired.name != name:
                raise NonDeterminismError(
                    f"history mismatch at event {fired.seq}: timer {name!r} is followed by "
                    f"{fired.type} {fired.name!r}"
                )
            self._pos += 1
            return
        wake_at = float(event.payload["wake_at"])
        if self._runtime.clock() >= wake_at:
            self._record("timer_fired", name)
            return
        raise Suspend(wake_at=wake_at)

    # --------------------------------------------------------------- signals

    def wait_signal(self, name: str, *, details: Any = None) -> Any:
        """Park the run until an external signal called ``name`` arrives.

        Returns the signal's payload. ``details`` is stored with the wait so an
        operator can see what is being asked for (see ``Runtime.pending``).
        """
        event = self._replay("signal_waiting", name)
        if event is None:
            self._record("signal_waiting", name, {"details": to_json(details)})
        received = self._peek()
        if received is not None:
            if received.type != "signal_received" or received.name != name:
                raise NonDeterminismError(
                    f"history mismatch at event {received.seq}: signal wait {name!r} is "
                    f"followed by {received.type} {received.name!r}"
                )
            self._pos += 1
            return received.payload["payload"]
        signal = self._journal.next_signal(self.run_id, name)
        if signal is None:
            raise Suspend(waiting_on=name)
        self._record(
            "signal_received", name, {"payload": signal.payload}, consume_signal=signal.id
        )
        return signal.payload

    # ------------------------------------------------------------- LLM, tools

    def llm(
        self,
        model: Model,
        messages: list[dict[str, Any]],
        tools: Sequence[Tool] | None = None,
        *,
        retry: RetryPolicy | None = None,
    ) -> dict[str, Any]:
        """Call a model durably.

        The response and its token usage and cost are journaled, so a resumed
        run never pays for the same completion twice and budgets are enforced
        identically on replay.
        """
        from .models import normalize_response

        self._check_llm_budget()
        specs = [t.spec() for t in tools or ()]

        def call() -> dict[str, Any]:
            response = normalize_response(model.complete(messages, specs))
            usage = response["usage"]
            usage["cost_usd"] = (
                usage["input_tokens"] * float(getattr(model, "input_price", 0.0))
                + usage["output_tokens"] * float(getattr(model, "output_price", 0.0))
            ) / 1_000_000
            return response

        result: dict[str, Any] = self._run_step(
            "llm",
            "llm",
            call,
            (),
            {},
            retry,
            fp_data=[messages, [s["name"] for s in specs]],
            meta={"model": model.name},
        )
        usage = result.get("usage", {})
        self.usage.llm_calls += 1
        self.usage.input_tokens += int(usage.get("input_tokens", 0))
        self.usage.output_tokens += int(usage.get("output_tokens", 0))
        self.usage.cost_usd += float(usage.get("cost_usd", 0.0))
        return result

    def tool(
        self, tool: Tool, arguments: dict[str, Any], *, retry: RetryPolicy | None = None
    ) -> Any:
        """Invoke a tool durably with the given arguments."""
        return self._run_step(
            "tool",
            f"tool:{tool.name}",
            tool.fn,
            (),
            dict(arguments),
            retry or tool.retry,
            fp_data=arguments,
        )

    def _check_llm_budget(self) -> None:
        budget, usage = self.budget, self.usage
        if budget.max_llm_calls is not None and usage.llm_calls >= budget.max_llm_calls:
            raise BudgetExceeded(f"LLM call budget of {budget.max_llm_calls} exhausted")
        if budget.max_tokens is not None and usage.total_tokens >= budget.max_tokens:
            raise BudgetExceeded(
                f"token budget of {budget.max_tokens} exhausted ({usage.total_tokens} used)"
            )
        if budget.max_cost_usd is not None and usage.cost_usd >= budget.max_cost_usd:
            raise BudgetExceeded(
                f"cost budget of ${budget.max_cost_usd:.4f} exhausted (${usage.cost_usd:.4f} spent)"
            )
