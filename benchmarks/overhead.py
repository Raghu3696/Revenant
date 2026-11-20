"""Measure what durability costs per step, and how fast replay is.

    python benchmarks/overhead.py [steps]
"""

from __future__ import annotations

import os
import platform
import sys
import tempfile
import time

from revenant import Runtime, workflow


def noop(i: int) -> int:
    return i


@workflow(name="bench.steps")
def bench(ctx, inp):
    total = 0
    for i in range(inp["steps"]):
        total += ctx.step("noop", noop, i)
    if inp["pause"]:
        ctx.wait_signal("go")
    return total


def measure(steps: int, durability: str) -> tuple[float, float]:
    with tempfile.TemporaryDirectory() as tmp:
        with Runtime(os.path.join(tmp, "bench.db"), durability=durability, heartbeat=False) as rt:
            began = time.perf_counter()
            first = rt.start(bench, {"steps": steps, "pause": True})
            live = time.perf_counter() - began
            rt.signal(first.run_id, "go")
            began = time.perf_counter()
            done = rt.resume(first.run_id)
            replay = time.perf_counter() - began
            assert done.result == steps * (steps - 1) // 2
    return live, replay


def main() -> None:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    print(f"{platform.python_implementation()} {platform.python_version()} on {platform.system()} "
          f"{platform.machine()}, {steps} steps (2 journal writes each)\n")
    print(f"{'durability':<12}{'live steps/s':>14}{'per step':>12}{'replay steps/s':>17}")
    for durability in ("normal", "full"):
        live, replay = measure(steps, durability)
        print(
            f"{durability:<12}{steps / live:>14,.0f}{live / steps * 1e6:>10,.0f}µs{steps / replay:>17,.0f}"
        )


if __name__ == "__main__":
    main()
