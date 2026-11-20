from __future__ import annotations

import io
import json

from revenant import RetryPolicy, ScriptedModel, run_agent, to_otlp, tool, workflow
from revenant.cli import main


@tool(requires_approval=True)
def ship(order: str) -> dict:
    """Ship an order."""
    return {"shipped": order}


MODEL = ScriptedModel(
    [{"tool_calls": [{"name": "ship", "arguments": {"order": "A1"}}]}, "Shipped."],
    name="demo-model",
    input_price=1.0,
    output_price=1.0,
)


@workflow(name="cli.agent")
def cli_agent(ctx, task):
    ctx.sleep(0, "warmup")
    return run_agent(ctx, MODEL, [ship], task)


def cli(db, *argv):
    out = io.StringIO()
    code = main(["--db", db, *argv], out=out)
    return code, out.getvalue()


def test_cli_end_to_end(rt, db, tmp_path):
    assert cli(db, "runs") == (0, "no runs\n")
    first = rt.start(cli_agent, "ship A1")
    short = first.run_id[:8]

    code, out = cli(db, "runs", "--status", "suspended")
    assert code == 0 and short in out and "approve:call_0_0" in out

    code, out = cli(db, "show", short)
    assert code == 0
    assert "status    suspended" in out and "waiting   signal approve:call_0_0" in out
    assert "model=demo-model" in out and "timer_fired" in out

    code, out = cli(db, "approve", short, "--reason", "ok")
    assert code == 0 and "approved: approve:call_0_0" in out and "status=completed" in out
    assert "Shipped." in out

    code, out = cli(db, "approve", short)
    assert code == 1 and "not waiting for an approval" in out

    code, out = cli(db, "show", short, "--json")
    doc = json.loads(out)
    assert doc["run"]["status"] == "completed" and doc["events"][-1]["type"] == "run_completed"

    code, out = cli(db, "fork", short, "--at", "6")
    assert code == 0 and "kept 6 event(s)" in out
    fork_id = out.split("-> ")[1].split()[0]
    code, out = cli(db, "resume", fork_id)
    assert code == 0 and "waiting for signal: approve:call_0_0" in out
    code, out = cli(db, "signal", fork_id, "approve:call_0_0", "--payload", '{"approved": false}')
    assert code == 0 and "status=completed" in out
    code, out = cli(db, "signal", fork_id, "x", "--payload", "{not json")
    assert code == 2

    target = tmp_path / "trace.json"
    code, out = cli(db, "export", short, "-o", str(target))
    assert code == 0 and json.loads(target.read_text())["resourceSpans"]

    assert cli(db, "show", "nope")[0] == 1
    assert cli(db, "-i", "no_such_module_xyz", "runs")[0] == 2
    code, out = cli(db, "worker", "--once")
    assert code == 0 and "drove 0 run(s)" in out


def test_cli_reports_unresolvable_workflow(rt, db):
    run_id = rt.submit("not.importable:wf")
    code, out = cli(db, "resume", run_id)
    assert code == 1 and "cannot resume here" in out


def test_otlp_export_structure(rt):
    @workflow(name="export.wf")
    def wf(ctx, task):
        ctx.step("flaky", _flaky, retry=RetryPolicy(max_attempts=2, initial_delay=0))
        try:
            ctx.step("broken", _broken, retry=RetryPolicy(max_attempts=1))
        except Exception:
            pass
        return run_agent(ctx, MODEL, [ship], task)

    _STATE.clear()
    first = rt.start(wf, "ship")
    done = rt.approve(first.run_id)
    run = rt.journal.get_run(done.run_id)
    doc = to_otlp(run, rt.journal.events(run.run_id), service_name="svc")

    resource = doc["resourceSpans"][0]
    assert resource["resource"]["attributes"] == [{"key": "service.name", "value": {"stringValue": "svc"}}]
    spans = resource["scopeSpans"][0]["spans"]
    root, children = spans[0], spans[1:]
    assert root["name"] == "run export.wf" and "parentSpanId" not in root
    assert len(root["traceId"]) == 32 and len(root["spanId"]) == 16
    assert all(s["traceId"] == root["traceId"] and s["parentSpanId"] == root["spanId"] for s in children)
    assert len({s["spanId"] for s in spans}) == len(spans)
    assert all(int(s["startTimeUnixNano"]) <= int(s["endTimeUnixNano"]) for s in spans)

    names = [s["name"] for s in children]
    assert names == [
        "flaky",
        "broken",
        "chat demo-model",
        "wait_signal approve:call_0_0",
        "execute_tool ship",
        "chat demo-model",
    ]

    def attrs(span):
        return {a["key"]: next(iter(a["value"].values())) for a in span["attributes"]}

    assert attrs(children[0])["revenant.step.attempts"] == "2"
    assert children[1]["status"]["code"] == 2 and attrs(children[1])["error.type"] == "ValueError"
    chat = attrs(children[2])
    assert chat["gen_ai.request.model"] == "demo-model" and int(chat["gen_ai.usage.input_tokens"]) > 0
    assert attrs(children[4])["gen_ai.tool.name"] == "ship"
    assert attrs(root)["revenant.run.status"] == "completed"


_STATE: dict = {}


def _flaky():
    if not _STATE.setdefault("failed", False):
        _STATE["failed"] = True
        raise ConnectionError("once")
    return "ok"


def _broken():
    raise ValueError("always")
