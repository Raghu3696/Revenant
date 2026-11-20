"""Retry policies, budgets and usage accounting."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

from .errors import NonRetryableError


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with deterministic jitter.

    Jitter is derived from the step's idempotency key instead of a random
    source, so a given run always backs off the same way. That keeps test runs
    and incident reproductions exact while still spreading load across runs.
    """

    max_attempts: int = 3
    initial_delay: float = 0.5
    backoff: float = 2.0
    max_delay: float = 30.0
    jitter: float = 0.1
    retry_on: tuple[type[BaseException], ...] = (Exception,)
    non_retryable: tuple[type[BaseException], ...] = field(default=(NonRetryableError,))

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    def should_retry(self, exc: BaseException, attempt: int) -> bool:
        if attempt >= self.max_attempts:
            return False
        if isinstance(exc, self.non_retryable):
            return False
        return isinstance(exc, self.retry_on)

    def delay(self, attempt: int, key: str = "") -> float:
        base = min(self.max_delay, self.initial_delay * self.backoff ** (attempt - 1))
        digest = hashlib.sha256(f"{key}:{attempt}".encode()).digest()
        unit = int.from_bytes(digest[:8], "big") / 2**64
        return max(0.0, base * (1.0 + self.jitter * unit))


NO_RETRY = RetryPolicy(max_attempts=1)


@dataclass(frozen=True)
class Budget:
    """Hard limits for one run. ``None`` means unlimited.

    Limits are checked *before* each operation starts, so a run can overshoot a
    token or cost limit by at most one model call, and never starts a call it
    is not allowed to make.
    """

    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_llm_calls: int | None = None
    max_steps: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Budget:
        return cls(**(data or {}))


@dataclass
class Usage:
    """Cumulative resource use of a run, rebuilt identically on every replay."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0
    steps: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["cost_usd"] = round(self.cost_usd, 8)
        data["total_tokens"] = self.total_tokens
        return data
