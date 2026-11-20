"""Tools and a durable tool-calling agent loop."""

from __future__ import annotations

import inspect
import json
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from .context import Context
from .errors import AgentError, StepFailed
from .models import Message, Model
from .policy import RetryPolicy

_JSON_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _schema_for(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        options = [a for a in get_args(annotation) if a is not type(None)]
        return _schema_for(options[0]) if len(options) == 1 else {}
    if origin is Literal:
        return {"enum": list(get_args(annotation))}
    if origin in (list, set, tuple, Sequence):
        args = get_args(annotation)
        schema: dict[str, Any] = {"type": "array"}
        if args and args[0] is not Ellipsis:
            schema["items"] = _schema_for(args[0])
        return schema
    if origin is dict:
        return {"type": "object"}
    if annotation in _JSON_TYPES:
        return {"type": _JSON_TYPES[annotation]}
    return {}


def _parameters_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in inspect.signature(fn).parameters.items():
        if name == "idempotency_key" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        properties[name] = _schema_for(hints.get(name, Any))
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class Tool:
    """A function an agent may call, with its JSON schema."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    requires_approval: bool = False
    retry: RetryPolicy | None = None

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    requires_approval: bool = False,
    retry: RetryPolicy | None = None,
) -> Any:
    """Turn a function into a :class:`Tool`.

    The JSON schema comes from the type hints and the description from the
    docstring. Declare an ``idempotency_key`` parameter to receive a key that
    stays the same across retries and crash recovery; it is hidden from the
    model. ``requires_approval=True`` pauses the run for a human decision
    before every call.
    """

    def build(func: Callable[..., Any]) -> Tool:
        return Tool(
            name=name or func.__name__,
            description=description or inspect.getdoc(func) or "",
            parameters=_parameters_schema(func),
            fn=func,
            requires_approval=requires_approval,
            retry=retry,
        )

    return build(fn) if fn is not None else build


def run_agent(
    ctx: Context,
    model: Model,
    tools: Sequence[Tool],
    task: str,
    *,
    system: str | None = None,
    max_turns: int = 12,
) -> dict[str, Any]:
    """Run a tool-calling loop to completion, durably.

    Every model call and tool call is a journaled step, so the loop survives a
    crash at any point without repeating finished work. Tools marked
    ``requires_approval`` suspend the run on the signal ``approve:<call id>``
    until a human answers with ``{"approved": bool, "reason": str | None}``.

    Returns ``{"output": str | None, "turns": int, "tool_calls": int}``.
    """
    by_name = {t.name: t for t in tools}
    messages: list[Message] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": task})

    executed = 0
    for turn in range(1, max_turns + 1):
        reply = ctx.llm(model, messages, tools)
        messages.append(
            {"role": "assistant", "content": reply["content"], "tool_calls": reply["tool_calls"]}
        )
        if not reply["tool_calls"]:
            return {"output": reply["content"], "turns": turn, "tool_calls": executed}

        for call in reply["tool_calls"]:
            target = by_name.get(call["name"])
            if target is None:
                outcome: Any = {"error": f"unknown tool {call['name']!r}"}
            else:
                outcome = None
                if target.requires_approval:
                    decision = ctx.wait_signal(
                        f"approve:{call['id']}",
                        details={"tool": target.name, "arguments": call["arguments"]},
                    )
                    if not (isinstance(decision, dict) and decision.get("approved")):
                        reason = decision.get("reason") if isinstance(decision, dict) else None
                        outcome = {"error": "denied by human reviewer", "reason": reason}
                if outcome is None:
                    try:
                        outcome = ctx.tool(target, call["arguments"])
                        executed += 1
                    except StepFailed as exc:
                        outcome = {"error": f"{exc.error['type']}: {exc.error['message']}"}
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["name"],
                    "content": json.dumps(outcome),
                }
            )
    raise AgentError(f"agent did not finish within {max_turns} turns")
