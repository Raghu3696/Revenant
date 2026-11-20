"""Kill a worker with SIGKILL in the middle of a pipeline, then watch another
process finish the job without repeating completed work.

    python examples/crash_recovery.py
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time

from revenant import Runtime, workflow

RUN_ID = "invoice-run"


def effect(name: str, log_path: str, idempotency_key: str) -> str:
    """Stands in for a real side effect: an API call, a payment, an email."""
    crash = name == os.environ.get("REVENANT_DEMO_CRASH_ON")
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"{name}\t{idempotency_key}\n")
    print(f"    [pid {os.getpid()}] executing {name}", flush=True)
    if crash:
        print(f"    [pid {os.getpid()}] power cut: SIGKILL during {name}", flush=True)
        os.kill(os.getpid(), signal.SIGKILL)
    return f"{name}: done"


@workflow(name="examples.invoice_pipeline")
def invoice_pipeline(ctx, inp):
    results = []
    for name in ("fetch-orders", "compute-totals", "charge-card", "send-receipt"):
        results.append(ctx.step(name, effect, name, inp["log"]))
    return results


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        with Runtime(sys.argv[2], lease_ttl=1.0) as rt:
            rt.start(invoice_pipeline, {"log": sys.argv[3]}, run_id=RUN_ID)
        return

    workdir = tempfile.mkdtemp(prefix="revenant-demo-")
    db, log = os.path.join(workdir, "journal.db"), os.path.join(workdir, "effects.log")

    print("1. Worker A starts the pipeline and is killed mid-step:")
    child = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", db, log],
        env={**os.environ, "REVENANT_DEMO_CRASH_ON": "charge-card"},
        check=False,
    )
    print(f"    worker A exit code: {child.returncode} (killed)\n")

    print("2. Worker B polls the journal. Once A's lease lapses it takes the run over:")
    with Runtime(db, lease_ttl=1.0) as rt:
        deadline = time.monotonic() + 30
        while (run := rt.journal.get_run(RUN_ID)).status in ("pending", "running"):
            if time.monotonic() > deadline:
                raise SystemExit("recovery timed out")
            rt.run_worker(stop_when_idle=True)
            time.sleep(0.1)
        if run.status != "completed":
            raise SystemExit(f"run ended as {run.status}: {run.error}")

    print(f"\n3. Result: {run.result}\n")
    print("4. Side effects that actually happened (step, idempotency key):")
    with open(log, encoding="utf-8") as handle:
        for line in handle:
            step, key = line.rstrip("\n").split("\t")
            print(f"    {step:<16}{key}")
    print(
        "\n   fetch-orders and compute-totals ran once. charge-card was in flight when A died,\n"
        "   so B ran it again under the SAME idempotency key, which is what lets a payment\n"
        "   API deduplicate it. Nothing that had finished was repeated.\n"
    )
    print(f"   Inspect the journal:  revenant --db {db} show {RUN_ID}")


if __name__ == "__main__":
    main()
