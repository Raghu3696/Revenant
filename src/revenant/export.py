"""Export a run's journal as an OpenTelemetry (OTLP/JSON) trace.

The output follows the OTLP JSON encoding of ``ExportTraceServiceRequest`` and
uses the OpenTelemetry GenAI semantic-convention attribute names for model and
tool spans, so it can be posted to an OTLP/HTTP collector endpoint as-is.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .journal import Event, RunRecord

_SPAN_KIND_INTERNAL = 1
_SPAN_KIND_CLIENT = 3
_STATUS_OK = 1
_STATUS_ERROR = 2


def _hex_id(text: str, length: int) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:length]


def _nanos(ts: float) -> str:
    return str(int(ts * 1_000_000_000))


def _attribute(key: str, value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        encoded: dict[str, Any] = {"boolValue": value}
    elif isinstance(value, int):
        encoded = {"intValue": str(value)}
    elif isinstance(value, float):
        encoded = {"doubleValue": value}
    else:
        encoded = {"stringValue": str(value)}
    return {"key": key, "value": encoded}


def _attributes(values: dict[str, Any]) -> list[dict[str, Any]]:
    return [_attribute(k, v) for k, v in values.items() if v is not None]


def to_otlp(run: RunRecord, events: list[Event], *, service_name: str = "revenant") -> dict[str, Any]:
    """Build an OTLP/JSON trace for one run."""
    trace_id = _hex_id(run.run_id, 32)
    root_id = _hex_id(f"{run.run_id}:root", 16)
    start = events[0].ts if events else run.created_at
    end = events[-1].ts if events else run.updated_at

    def span(
        seq: int,
        name: str,
        begin: float,
        finish: float,
        attributes: dict[str, Any],
        *,
        kind: int = _SPAN_KIND_INTERNAL,
        error: str | None = None,
    ) -> dict[str, Any]:
        status: dict[str, Any] = {"code": _STATUS_ERROR, "message": error} if error else {"code": _STATUS_OK}
        return {
            "traceId": trace_id,
            "spanId": _hex_id(f"{run.run_id}:{seq}", 16),
            "parentSpanId": root_id,
            "name": name,
            "kind": kind,
            "startTimeUnixNano": _nanos(begin),
            "endTimeUnixNano": _nanos(finish),
            "attributes": _attributes(attributes),
            "status": status,
        }

    spans: list[dict[str, Any]] = []
    index = 0
    while index < len(events):
        event = events[index]
        index += 1
        if event.type == "step_started":
            kind = event.payload.get("kind", "step")
            attempts = 1
            finish = end
            outcome: Event | None = None
            while index < len(events) and events[index].type in (
                "step_attempt_failed",
                "step_completed",
                "step_failed",
            ):
                follower = events[index]
                index += 1
                if follower.type == "step_attempt_failed":
                    attempts += 1
                    continue
                outcome, finish = follower, follower.ts
                break
            attributes: dict[str, Any] = {
                "revenant.step.kind": kind,
                "revenant.step.seq": event.seq,
                "revenant.step.attempts": attempts,
                "revenant.step.in_flight": outcome is None,
            }
            error = None
            if outcome is not None and outcome.type == "step_failed":
                err = outcome.payload["error"]
                error = f"{err.get('type')}: {err.get('message')}"
                attributes["error.type"] = err.get("type")
            name = event.name
            span_kind = _SPAN_KIND_INTERNAL
            if kind == "llm":
                model = (event.payload.get("meta") or {}).get("model")
                name = f"chat {model}" if model else "chat"
                span_kind = _SPAN_KIND_CLIENT
                attributes["gen_ai.operation.name"] = "chat"
                attributes["gen_ai.request.model"] = model
                if outcome is not None and outcome.type == "step_completed":
                    usage = outcome.payload["result"].get("usage", {})
                    attributes["gen_ai.usage.input_tokens"] = usage.get("input_tokens")
                    attributes["gen_ai.usage.output_tokens"] = usage.get("output_tokens")
                    attributes["revenant.cost_usd"] = float(usage.get("cost_usd", 0.0))
            elif kind == "tool":
                tool_name = event.name.removeprefix("tool:")
                name = f"execute_tool {tool_name}"
                attributes["gen_ai.operation.name"] = "execute_tool"
                attributes["gen_ai.tool.name"] = tool_name
            spans.append(span(event.seq, name, event.ts, finish, attributes, kind=span_kind, error=error))
        elif event.type in ("timer_started", "signal_waiting"):
            closing = "timer_fired" if event.type == "timer_started" else "signal_received"
            finish = end
            if index < len(events) and events[index].type == closing:
                finish = events[index].ts
                index += 1
            label = "timer" if event.type == "timer_started" else "wait_signal"
            spans.append(
                span(
                    event.seq,
                    f"{label} {event.name}",
                    event.ts,
                    finish,
                    {"revenant.step.kind": label, "revenant.step.seq": event.seq},
                )
            )

    root_error = None
    if run.status == "failed" and run.error:
        root_error = f"{run.error.get('type')}: {run.error.get('message')}"
    usage = run.usage or {}
    root = {
        "traceId": trace_id,
        "spanId": root_id,
        "name": f"run {run.workflow}",
        "kind": _SPAN_KIND_INTERNAL,
        "startTimeUnixNano": _nanos(start),
        "endTimeUnixNano": _nanos(end),
        "attributes": _attributes(
            {
                "revenant.run.id": run.run_id,
                "revenant.run.status": run.status,
                "revenant.run.workflow": run.workflow,
                "revenant.run.parent_id": run.parent_run_id,
                "gen_ai.usage.input_tokens": usage.get("input_tokens"),
                "gen_ai.usage.output_tokens": usage.get("output_tokens"),
                "revenant.cost_usd": usage.get("cost_usd"),
            }
        ),
        "status": {"code": _STATUS_ERROR, "message": root_error} if root_error else {"code": _STATUS_OK},
    }
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": _attributes({"service.name": service_name})},
                "scopeSpans": [{"scope": {"name": "revenant"}, "spans": [root, *spans]}],
            }
        ]
    }
