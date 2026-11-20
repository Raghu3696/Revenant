from __future__ import annotations

import pytest

from revenant import Runtime


class FakeClock:
    """Manually advanced clock; sleeping advances it instead of blocking."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


class SimulatedCrash(BaseException):
    """Stands in for the process dying at an arbitrary point."""


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def db(tmp_path) -> str:
    return str(tmp_path / "journal.db")


@pytest.fixture
def make_runtime(db, clock):
    created: list[Runtime] = []

    def factory(**kwargs) -> Runtime:
        kwargs.setdefault("clock", clock)
        kwargs.setdefault("sleeper", clock.sleep)
        kwargs.setdefault("heartbeat", False)
        rt = Runtime(db, **kwargs)
        created.append(rt)
        return rt

    yield factory
    for rt in created:
        rt.close()


@pytest.fixture
def rt(make_runtime) -> Runtime:
    return make_runtime(worker_id="worker-a")
