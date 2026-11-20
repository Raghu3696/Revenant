from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar, Literal

import pytest

from revenant import (
    AgentError,
    ModelError,
    NonRetryableError,
    OpenAICompatibleModel,
    RetryPolicy,
    ScriptedModel,
    run_agent,
    tool,
    workflow,
)


def test_tool_schema_from_type_hints():
    @tool(requires_approval=True)
    def transfer(
        account: str,
        amount: float,
        currency: Literal["USD", "EUR"] = "USD",
        tags: list[str] | None = None,
        dry_run: bool = False,
        idempotency_key: str = "",
    ) -> dict:
        """Move money between accounts."""
        return {}

    assert transfer.name == "transfer" and transfer.requires_approval
    assert transfer.description == "Move money between accounts."
    assert transfer.parameters == {
        "type": "object",
        "properties": {
            "account": {"type": "string"},
            "amount": {"type": "number"},
            "currency": {"enum": ["USD", "EUR"]},
            "tags": {"type": "array", "items": {"type": "string"}},
            "dry_run": {"type": "boolean"},
        },
        "required": ["account", "amount"],
        "additionalProperties": False,
    }


def make_agent(name, model, tools, **kwargs):
    @workflow(name=name)
    def wf(ctx, task):
        return run_agent(ctx, model, tools, task, system="You are helpful.", **kwargs)

    return wf


def test_agent_loop_with_approval_gate_survives_restart(rt, make_runtime):
    executed = []

    @tool
    def lookup(order_id: str) -> dict:
        """Look up an order."""
        executed.append(("lookup", order_id))
        return {"order_id": order_id, "total": 42.5}

    @tool(requires_approval=True)
    def refund(order_id: str, amount: float, idempotency_key: str = "") -> dict:
        """Refund an order."""
        executed.append(("refund", order_id, amount, idempotency_key))
        return {"refunded": amount}

    model = ScriptedModel(
        [
            {"tool_calls": [{"name": "lookup", "arguments": {"order_id": "A1"}}]},
            {"content": "Refunding.", "tool_calls": [{"name": "refund", "arguments": {"order_id": "A1", "amount": 42.5}}]},
            "Refund of 42.5 issued.",
        ]
    )
    wf = make_agent("agent.approval", model, [lookup, refund])

    first = rt.start(wf, "Refund order A1")
    assert first.suspended and first.waiting_on == "approve:call_1_0"
    assert rt.pending(first.run_id) == {
        "signal": "approve:call_1_0",
        "details": {"tool": "refund", "arguments": {"order_id": "A1", "amount": 42.5}},
    }
    assert executed == [("lookup", "A1")]  # nothing risky ran yet

    other = make_runtime(worker_id="worker-b")  # approval arrives in another process
    done = other.approve(first.run_id, reason="verified")
    assert done.result == {"output": "Refund of 42.5 issued.", "turns": 3, "tool_calls": 2}
    assert executed[1][:3] == ("refund", "A1", 42.5) and executed[1][3].startswith(first.run_id)
    assert len(executed) == 2
    assert done.usage["llm_calls"] == 3 and done.usage["total_tokens"] > 0


def test_denied_tool_is_not_executed_and_fork_explores_the_other_branch(rt):
    executed = []

    @tool(requires_approval=True)
    def delete_account(user: str) -> str:
        """Delete a user account."""
        executed.append(user)
        return "deleted"

    def script(messages, tools):
        last = messages[-1]
        if last["role"] == "user":
            return {"tool_calls": [{"name": "delete_account", "arguments": {"user": "bob"}}]}
        return "Denied, account kept." if "denied" in last["content"] else "Account deleted."

    wf = make_agent("agent.denied", ScriptedModel(script), [delete_account])
    first = rt.start(wf, "delete bob")
    denied = rt.approve(first.run_id, approved=False, reason="too risky")
    assert denied.result["output"] == "Denied, account kept." and executed == []

    # Time travel: branch from just before the decision and approve instead.
    events = rt.journal.events(first.run_id)
    waiting = next(e for e in events if e.type == "signal_waiting")
    fork_id = rt.fork(first.run_id, at_seq=waiting.seq + 1)
    assert rt.resume(fork_id).suspended
    approved = rt.approve(fork_id)
    assert approved.result["output"] == "Account deleted." and executed == ["bob"]
    assert rt.journal.get_run(first.run_id).result["output"] == "Denied, account kept."


def test_unknown_tool_and_failing_tool_are_reported_to_the_model(rt):
    @tool(retry=RetryPolicy(max_attempts=2, initial_delay=0))
    def unstable() -> str:
        """Always fails."""
        raise RuntimeError("backend down")

    seen = []

    def script(messages, tools):
        turn = sum(1 for m in messages if m["role"] == "assistant")
        if turn == 0:
            return {"tool_calls": [{"name": "ghost", "arguments": {}}, {"name": "unstable", "arguments": {}}]}
        seen.extend(json.loads(m["content"]) for m in messages if m["role"] == "tool")
        return "gave up"

    result = rt.start(make_agent("agent.errors", ScriptedModel(script), [unstable]), "go")
    assert result.result["output"] == "gave up" and result.result["tool_calls"] == 0
    assert seen == [{"error": "unknown tool 'ghost'"}, {"error": "RuntimeError: backend down"}]


def test_agent_stops_after_max_turns(rt):
    @tool
    def noop() -> str:
        """Do nothing."""
        return "ok"

    looping = ScriptedModel(lambda m, t: {"tool_calls": [{"name": "noop", "arguments": {}}]})
    result = rt.start(make_agent("agent.loop", looping, [noop], max_turns=3), "loop")
    assert result.status == "failed" and result.error["type"] == "AgentError"
    assert AgentError.__name__ in result.error["traceback"]


def test_scripted_model_running_out_of_script_fails_fast(rt):
    result = rt.start(make_agent("agent.short", ScriptedModel([]), []), "hi")
    assert result.status == "failed" and "scripted model" in result.error["message"]


# --------------------------------------------------------------- HTTP adapter


class _Handler(BaseHTTPRequestHandler):
    replies: ClassVar[list] = []
    requests: ClassVar[list] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        status, payload = type(self).replies.pop(0)
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    _Handler.replies, _Handler.requests = [], []
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1", _Handler
    httpd.shutdown()
    httpd.server_close()


def _completion(message, prompt=11, completion=7):
    return {"choices": [{"message": message}], "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


def test_openai_compatible_adapter_round_trip(rt, server):
    base_url, handler = server
    handler.replies = [
        (503, {"error": "overloaded"}),  # retried by the durable step
        (
            200,
            _completion(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "tc_1", "type": "function", "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'}}
                    ],
                }
            ),
        ),
        (200, _completion({"role": "assistant", "content": "The answer is 5."}, 30, 6)),
    ]

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    model = OpenAICompatibleModel("test-model", base_url=base_url, api_key="sk-test", input_price=1.0, output_price=2.0)

    @workflow(name="agent.http")
    def wf(ctx, task):
        return run_agent(ctx, model, [add], task)

    rt.default_retry = RetryPolicy(max_attempts=3, initial_delay=0)
    result = rt.start(wf, "What is 2 + 3?")
    assert result.result == {"output": "The answer is 5.", "turns": 2, "tool_calls": 1}
    assert result.usage["input_tokens"] == 41 and result.usage["output_tokens"] == 13
    assert result.usage["cost_usd"] == pytest.approx((41 * 1.0 + 13 * 2.0) / 1e6)

    first, _, last = handler.requests
    assert first["path"] == "/v1/chat/completions" and first["auth"] == "Bearer sk-test"
    assert first["body"]["model"] == "test-model"
    assert first["body"]["tools"][0] == {"type": "function", "function": add.spec()}
    # the follow-up request carries the tool call and its result in OpenAI wire format
    assistant, tool_msg = last["body"]["messages"][-2:]
    assert assistant["tool_calls"][0] == {
        "id": "tc_1",
        "type": "function",
        "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
    }
    assert tool_msg == {"role": "tool", "tool_call_id": "tc_1", "content": "5"}


def test_openai_adapter_error_mapping(server):
    base_url, handler = server
    model = OpenAICompatibleModel("m", base_url=base_url, api_key="")
    handler.replies = [(429, {"error": "slow down"}), (401, {"error": "bad key"}), (200, {"unexpected": True})]
    with pytest.raises(ModelError, match="HTTP 429"):
        model.complete([{"role": "user", "content": "x"}], [])
    with pytest.raises(NonRetryableError, match="HTTP 401"):
        model.complete([{"role": "user", "content": "x"}], [])
    with pytest.raises(ModelError, match="malformed"):
        model.complete([{"role": "user", "content": "x"}], [])
    assert handler.requests[0]["auth"] is None

    unreachable = OpenAICompatibleModel("m", base_url="http://127.0.0.1:9/v1", timeout=2)
    with pytest.raises(ModelError, match="failed"):
        unreachable.complete([{"role": "user", "content": "x"}], [])
