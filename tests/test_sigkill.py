"""End-to-end crash recovery across real OS processes killed with SIGKILL."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="needs POSIX signals")

SCRIPT = textwrap.dedent(
    """
    import json, os, signal, sys
    from revenant import Runtime, workflow

    db, log_path, mode = sys.argv[1], sys.argv[2], sys.argv[3]

    def effect(name, idempotency_key):
        with open(log_path, "a") as fh:
            fh.write(json.dumps({"step": name, "key": idempotency_key}) + "\\n")
            fh.flush()
            os.fsync(fh.fileno())
        if name == "third" and mode == "crash":
            os.kill(os.getpid(), signal.SIGKILL)  # die mid-step, no cleanup at all
        return name.upper()

    @workflow(name="sigkill.pipeline")
    def pipeline(ctx, inp):
        return [ctx.step(n, effect, n) for n in ("first", "second", "third", "fourth")]

    rt = Runtime(db, lease_ttl=0.5)
    if mode == "crash":
        rt.start(pipeline, run_id="the-run")
    else:
        import time
        deadline = time.time() + 20
        while time.time() < deadline:
            rt.run_worker(stop_when_idle=True)  # waits for the dead worker's lease to lapse
            run = rt.journal.get_run("the-run")
            if run.status == "completed":
                print(json.dumps(run.result))
                break
            time.sleep(0.1)
    """
)


def test_process_killed_mid_step_is_recovered_by_another_process(tmp_path):
    script = tmp_path / "pipeline.py"
    script.write_text(SCRIPT)
    db, log = str(tmp_path / "j.db"), str(tmp_path / "effects.log")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}

    crashed = subprocess.run([sys.executable, str(script), db, log, "crash"], env=env, timeout=60)
    assert crashed.returncode == -signal.SIGKILL

    recovered = subprocess.run(
        [sys.executable, str(script), db, log, "recover"],
        env=env,
        timeout=60,
        capture_output=True,
        text=True,
    )
    assert recovered.returncode == 0, recovered.stderr
    assert json.loads(recovered.stdout) == ["FIRST", "SECOND", "THIRD", "FOURTH"]

    effects = [json.loads(line) for line in open(log)]
    steps = [e["step"] for e in effects]
    # first and second finished before the kill and were NOT re-executed; third
    # was in flight, so it ran again under the same idempotency key.
    assert steps == ["first", "second", "third", "third", "fourth"]
    third_keys = {e["key"] for e in effects if e["step"] == "third"}
    assert len(third_keys) == 1
