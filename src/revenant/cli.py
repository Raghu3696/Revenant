"""Command-line interface: inspect, resume, approve, fork and export runs."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from collections.abc import Sequence
from typing import Any

from . import __version__
from .errors import LeaseHeld, NonDeterminismError, RevenantError, WorkflowNotFound
from .export import to_otlp
from .journal import Event, RunRecord
from .runtime import RunResult, Runtime


def _short(value: Any, width: int = 88) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _stamp(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _describe(event: Event) -> str:
    payload = event.payload
    kind = event.type
    if kind == "run_started":
        return f"input={_short(payload.get('input'))}"
    if kind == "step_started":
        model = (payload.get("meta") or {}).get("model")
        return f"{event.name}" + (f"  model={model}" if model else "")
    if kind == "step_completed":
        result = payload.get("result")
        if isinstance(result, dict) and "usage" in result and "tool_calls" in result:
            usage = result["usage"]
            calls = ", ".join(c["name"] for c in result["tool_calls"]) or "-"
            return (
                f"{event.name}  tokens={usage.get('input_tokens', 0)}+{usage.get('output_tokens', 0)}"
                f"  tool_calls={calls}  content={_short(result.get('content'), 48)}"
            )
        return f"{event.name}  -> {_short(result)}"
    if kind == "step_attempt_failed":
        err = payload["error"]
        return f"{event.name}  attempt {payload['attempt']}: {err['type']}: {_short(err['message'], 60)}"
    if kind == "step_failed":
        err = payload["error"]
        return f"{event.name}  after {payload['attempts']} attempt(s): {err['type']}: {_short(err['message'], 60)}"
    if kind == "timer_started":
        return f"{event.name}  wake_at={_stamp(payload['wake_at'])}"
    if kind == "signal_waiting":
        return f"{event.name}  details={_short(payload.get('details'))}"
    if kind == "signal_received":
        return f"{event.name}  payload={_short(payload.get('payload'))}"
    if kind == "run_completed":
        return f"result={_short(payload.get('result'))}"
    if kind == "run_failed":
        err = payload["error"]
        return f"{err['type']}: {_short(err['message'])}"
    return event.name


def _print_result(result: RunResult, out: Any) -> None:
    line = f"run {result.run_id}  status={result.status}"
    if result.status == "completed":
        line += f"\nresult: {_short(result.result, 400)}"
    elif result.status == "failed" and result.error:
        line += f"\nerror: {result.error['type']}: {result.error['message']}"
    elif result.waiting_on:
        line += f"\nwaiting for signal: {result.waiting_on}"
    elif result.wake_at is not None:
        line += f"\nsleeping until: {_stamp(result.wake_at)}"
    print(line, file=out)


def _try_resume(rt: Runtime, run_id: str, out: Any, *, force: bool = False) -> int:
    try:
        _print_result(rt.resume(run_id, force=force), out)
    except WorkflowNotFound as exc:
        print(f"delivered, but cannot resume here: {exc}", file=out)
        print("pass -i MODULE to import the workflow, or let a worker pick the run up", file=out)
        return 1
    except (LeaseHeld, NonDeterminismError) as exc:
        print(f"cannot resume: {exc}", file=out)
        return 1
    return 0


def _cmd_runs(rt: Runtime, args: argparse.Namespace, out: Any) -> int:
    runs = rt.journal.list_runs(status=args.status, limit=args.limit)
    if not runs:
        print("no runs", file=out)
        return 0
    print(f"{'RUN':<14}{'STATUS':<11}{'WORKFLOW':<34}{'UPDATED':<21}WAITING", file=out)
    for run in runs:
        waiting = run.waiting_on or (f"timer {_stamp(run.wake_at)}" if run.wake_at else "")
        print(
            f"{run.run_id[:12]:<14}{run.status:<11}{_short(run.workflow, 32):<34}"
            f"{_stamp(run.updated_at):<21}{waiting}",
            file=out,
        )
    return 0


def _cmd_show(rt: Runtime, args: argparse.Namespace, out: Any) -> int:
    run: RunRecord = rt.journal.find_run(args.run)
    events = rt.journal.events(run.run_id)
    if args.json:
        doc = {
            "run": {k: getattr(run, k) for k in run.__dataclass_fields__},
            "events": [
                {"seq": e.seq, "type": e.type, "name": e.name, "payload": e.payload, "ts": e.ts}
                for e in events
            ],
        }
        print(json.dumps(doc, indent=2, sort_keys=True), file=out)
        return 0
    print(f"run       {run.run_id}", file=out)
    print(f"workflow  {run.workflow}", file=out)
    print(f"status    {run.status}", file=out)
    if run.parent_run_id:
        print(f"forked    from {run.parent_run_id} at event {run.forked_at_seq}", file=out)
    if run.usage:
        u = run.usage
        print(
            f"usage     {u.get('llm_calls', 0)} llm call(s), {u.get('total_tokens', 0)} tokens, "
            f"${u.get('cost_usd', 0.0):.6f}, {u.get('steps', 0)} step(s)",
            file=out,
        )
    pending = rt.pending(run.run_id)
    if pending and "signal" in pending:
        print(f"waiting   signal {pending['signal']}  {_short(pending['details'])}", file=out)
    elif pending:
        print(f"waiting   timer until {_stamp(pending['wake_at'])}", file=out)
    if run.error and run.status in ("failed", "stalled"):
        print(f"error     {run.error['type']}: {run.error['message']}", file=out)
    print("", file=out)
    origin = events[0].ts if events else run.created_at
    for event in events:
        print(
            f"  #{event.seq:<4}+{event.ts - origin:8.3f}s  {event.type:<20}{_describe(event)}",
            file=out,
        )
    return 0


def _cmd_resume(rt: Runtime, args: argparse.Namespace, out: Any) -> int:
    run = rt.journal.find_run(args.run)
    return _try_resume(rt, run.run_id, out, force=args.force)


def _cmd_signal(rt: Runtime, args: argparse.Namespace, out: Any) -> int:
    run = rt.journal.find_run(args.run)
    try:
        payload = json.loads(args.payload) if args.payload is not None else None
    except json.JSONDecodeError as exc:
        print(f"--payload is not valid JSON: {exc}", file=out)
        return 2
    rt.signal(run.run_id, args.name, payload)
    print(f"signal {args.name!r} delivered to run {run.run_id}", file=out)
    return 0 if args.no_resume else _try_resume(rt, run.run_id, out)


def _cmd_approve(rt: Runtime, args: argparse.Namespace, out: Any) -> int:
    run = rt.journal.find_run(args.run)
    if run.status != "suspended" or not (run.waiting_on or "").startswith("approve:"):
        print(f"run {run.run_id} is not waiting for an approval", file=out)
        return 1
    assert run.waiting_on is not None
    rt.signal(run.run_id, run.waiting_on, {"approved": not args.deny, "reason": args.reason})
    print(f"{'denied' if args.deny else 'approved'}: {run.waiting_on}", file=out)
    return 0 if args.no_resume else _try_resume(rt, run.run_id, out)


def _cmd_fork(rt: Runtime, args: argparse.Namespace, out: Any) -> int:
    run = rt.journal.find_run(args.run)
    new_id = rt.fork(run.run_id, args.at)
    kept = len(rt.journal.events(new_id))
    print(f"forked run {run.run_id} -> {new_id} (kept {kept} event(s))", file=out)
    return _try_resume(rt, new_id, out) if args.resume else 0


def _cmd_export(rt: Runtime, args: argparse.Namespace, out: Any) -> int:
    run = rt.journal.find_run(args.run)
    doc = to_otlp(run, rt.journal.events(run.run_id), service_name=args.service)
    text = json.dumps(doc, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(f"wrote {args.output}", file=out)
    else:
        print(text, file=out)
    return 0


def _cmd_worker(rt: Runtime, args: argparse.Namespace, out: Any) -> int:
    print(f"worker {rt.worker_id} polling {rt.journal.path}", file=out)
    try:
        driven = rt.run_worker(poll_interval=args.poll, stop_when_idle=args.once)
    except KeyboardInterrupt:
        print("worker stopped", file=out)
        return 0
    print(f"drove {driven} run(s)", file=out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="revenant", description="Inspect and operate durable agent runs."
    )
    parser.add_argument("--version", action="version", version=f"revenant {__version__}")
    parser.add_argument(
        "--db",
        default=os.environ.get("REVENANT_DB", "revenant.db"),
        help="journal database (default: $REVENANT_DB or ./revenant.db)",
    )
    parser.add_argument(
        "-i",
        "--import",
        dest="imports",
        action="append",
        default=[],
        metavar="MODULE",
        help="import MODULE so its workflows are registered (repeatable)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("runs", help="list runs")
    p.add_argument("--status", choices=["pending", "running", "suspended", "completed", "failed", "stalled"])
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=_cmd_runs)

    p = sub.add_parser("show", help="show a run's timeline")
    p.add_argument("run")
    p.add_argument("--json", action="store_true", help="full machine-readable dump")
    p.set_defaults(func=_cmd_show)

    p = sub.add_parser("resume", help="continue a run from its journal")
    p.add_argument("run")
    p.add_argument("--force", action="store_true", help="take over the lease from a dead worker")
    p.set_defaults(func=_cmd_resume)

    p = sub.add_parser("signal", help="deliver a signal to a run")
    p.add_argument("run")
    p.add_argument("name")
    p.add_argument("--payload", help="JSON payload")
    p.add_argument("--no-resume", action="store_true")
    p.set_defaults(func=_cmd_signal)

    p = sub.add_parser("approve", help="answer the approval a run is waiting on")
    p.add_argument("run")
    p.add_argument("--deny", action="store_true")
    p.add_argument("--reason")
    p.add_argument("--no-resume", action="store_true")
    p.set_defaults(func=_cmd_approve)

    p = sub.add_parser("fork", help="branch a new run from an earlier point")
    p.add_argument("run")
    p.add_argument("--at", type=int, help="keep events with seq < AT (default: all but the outcome)")
    p.add_argument("--resume", action="store_true", help="drive the fork immediately")
    p.set_defaults(func=_cmd_fork)

    p = sub.add_parser("export", help="export a run as an OpenTelemetry (OTLP/JSON) trace")
    p.add_argument("run")
    p.add_argument("-o", "--output")
    p.add_argument("--service", default="revenant")
    p.set_defaults(func=_cmd_export)

    p = sub.add_parser("worker", help="drive due runs: new, timers, signals, crashed")
    p.add_argument("--poll", type=float, default=0.5)
    p.add_argument("--once", action="store_true", help="exit when nothing is due")
    p.set_defaults(func=_cmd_worker)
    return parser


def main(argv: Sequence[str] | None = None, out: Any = None) -> int:
    out = out or sys.stdout
    args = build_parser().parse_args(argv)
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    try:
        for module in args.imports:
            importlib.import_module(module)
    except ImportError as exc:
        print(f"error: cannot import {exc.name or 'module'}: {exc}", file=out)
        return 2
    rt = Runtime(args.db)
    try:
        return int(args.func(rt, args, out))
    except RevenantError as exc:
        print(f"error: {exc}", file=out)
        return 1
    finally:
        rt.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
