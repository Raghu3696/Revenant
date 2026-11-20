"""Exception hierarchy for Revenant."""

from __future__ import annotations

from typing import Any


class RevenantError(Exception):
    """Base class for all errors raised by Revenant."""


class NonDeterminismError(RevenantError):
    """The workflow code no longer matches the recorded history.

    Raised during replay when the workflow asks for an operation that differs
    from the one journaled at the same position. The run is parked as
    ``stalled`` (never marked failed) so it can be resumed once the code is
    fixed, or forked from an earlier point.
    """


class BudgetExceeded(RevenantError):
    """A token, cost, step or LLM-call budget was exhausted."""


class NonRetryableError(RevenantError):
    """Raise from a step to fail it immediately, skipping remaining retries."""


class ModelError(RevenantError):
    """A model provider call failed in a way that is worth retrying."""


class AgentError(RevenantError):
    """The agent loop could not finish (for example, it ran out of turns)."""


class WorkflowNotFound(RevenantError):
    """No workflow is registered or importable under the requested name."""


class RunNotFound(RevenantError):
    """No run exists with the requested id."""


class LeaseHeld(RevenantError):
    """Another live worker currently owns the run."""


class StepFailed(RevenantError):
    """A step exhausted its retries (or failed with a non-retryable error).

    The failure is journaled, so replay raises the *same* exception at the
    same point without re-running the step. Workflows may catch it.
    """

    def __init__(self, step: str, error: dict[str, Any], attempts: int) -> None:
        self.step = step
        self.error = error
        self.attempts = attempts
        super().__init__(
            f"step {step!r} failed after {attempts} attempt(s): "
            f"{error.get('type', 'Error')}: {error.get('message', '')}"
        )


class LeaseLost(BaseException):
    """This worker was fenced off: another worker took over the run.

    Derives from ``BaseException`` so a broad ``except Exception`` inside
    workflow code cannot swallow it and keep writing to a run it no longer owns.
    """


class Suspend(BaseException):
    """Internal control flow: the run is parked until a timer or a signal."""

    def __init__(self, *, wake_at: float | None = None, waiting_on: str | None = None) -> None:
        self.wake_at = wake_at
        self.waiting_on = waiting_on
        super().__init__("run suspended")
