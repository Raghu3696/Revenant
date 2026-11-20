"""Model providers.

A model is anything with a ``name`` and a ``complete(messages, tools)`` method
that returns a response dict. Models must be **stateless**: after a crash the
runtime replays earlier completions from the journal and only calls the model
for the turns that never finished, so a provider cannot rely on having seen
the earlier calls.

Message format (provider-neutral)::

    {"role": "system" | "user", "content": str}
    {"role": "assistant", "content": str | None, "tool_calls": [ToolCall, ...]}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}

    ToolCall = {"id": str, "name": str, "arguments": dict}
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from .errors import ModelError, NonRetryableError

Message = dict[str, Any]
Script = Sequence[Any] | Callable[[list[Message], list[dict[str, Any]]], Any]


@runtime_checkable
class Model(Protocol):
    name: str

    def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> Any: ...


def estimate_tokens(text: str) -> int:
    """Rough token estimate (about four characters per token)."""
    return max(1, len(text) // 4)


def normalize_response(response: Any) -> dict[str, Any]:
    """Coerce a provider response into the canonical shape."""
    if isinstance(response, str):
        response = {"content": response}
    if not isinstance(response, dict):
        raise NonRetryableError(f"model returned {type(response).__name__}, expected str or dict")
    calls = []
    for index, call in enumerate(response.get("tool_calls") or []):
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise NonRetryableError("tool call arguments must be an object")
        calls.append(
            {
                "id": str(call.get("id") or f"call_{index}"),
                "name": str(call["name"]),
                "arguments": arguments,
            }
        )
    usage = response.get("usage") or {}
    return {
        "content": response.get("content"),
        "tool_calls": calls,
        "usage": {
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
        },
    }


class ScriptedModel:
    """Deterministic model for tests, demos and offline development.

    ``script`` is either a list of responses, indexed by how many assistant
    turns the conversation already contains, or a function
    ``(messages, tools) -> response``. Both forms derive the reply from the
    conversation alone, which keeps the model stateless and replay-safe.
    """

    def __init__(
        self,
        script: Script,
        *,
        name: str = "scripted",
        input_price: float = 0.0,
        output_price: float = 0.0,
    ) -> None:
        self.script = script
        self.name = name
        self.input_price = input_price
        self.output_price = output_price

    def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> dict[str, Any]:
        turn = sum(1 for m in messages if m.get("role") == "assistant")
        if callable(self.script):
            raw = self.script(messages, tools)
        else:
            if turn >= len(self.script):
                raise NonRetryableError(
                    f"scripted model has {len(self.script)} response(s) but turn {turn} was requested"
                )
            raw = self.script[turn]
        if isinstance(raw, str):
            raw = {"content": raw}
        response = dict(raw)
        response["tool_calls"] = [
            {"id": call.get("id") or f"call_{turn}_{i}", **{k: v for k, v in call.items() if k != "id"}}
            for i, call in enumerate(response.get("tool_calls") or [])
        ]
        if "usage" not in response:
            output = json.dumps([response.get("content"), response["tool_calls"]])
            response["usage"] = {
                "input_tokens": estimate_tokens(json.dumps(messages)),
                "output_tokens": estimate_tokens(output),
            }
        return response


class OpenAICompatibleModel:
    """Chat Completions client for any OpenAI-compatible endpoint.

    Works with hosted APIs and local servers that expose
    ``POST {base_url}/chat/completions``. Uses only the standard library.
    Prices are USD per million tokens and feed cost budgets.
    """

    RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        input_price: float = 0.0,
        output_price: float = 0.0,
        timeout: float = 60.0,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.name = model
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.input_price = input_price
        self.output_price = output_price
        self.timeout = timeout
        self.extra_body = extra_body or {}

    def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.name,
            "messages": [self._encode(m) for m in messages],
            **self.extra_body,
        }
        if tools:
            body["tools"] = [{"type": "function", "function": spec} for spec in tools]
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as reply:
                data = json.loads(reply.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            message = f"HTTP {exc.code} from {self.base_url}: {detail}"
            if exc.code in self.RETRYABLE_STATUS:
                raise ModelError(message) from exc
            raise NonRetryableError(message) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ModelError(f"request to {self.base_url} failed: {exc}") from exc
        return self._decode(data)

    @staticmethod
    def _encode(message: Message) -> dict[str, Any]:
        role = message["role"]
        if role == "assistant":
            encoded: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
            calls = message.get("tool_calls") or []
            if calls:
                encoded["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["arguments"]),
                        },
                    }
                    for call in calls
                ]
            return encoded
        if role == "tool":
            return {
                "role": "tool",
                "tool_call_id": message["tool_call_id"],
                "content": message["content"],
            }
        return {"role": role, "content": message["content"]}

    @staticmethod
    def _decode(data: dict[str, Any]) -> dict[str, Any]:
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelError(f"malformed completion response: {str(data)[:300]}") from exc
        calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            raw = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw)
            except json.JSONDecodeError:
                arguments = {"_raw": raw}
            if not isinstance(arguments, dict):
                arguments = {"_raw": raw}
            calls.append({"id": call.get("id"), "name": function.get("name", ""), "arguments": arguments})
        usage = data.get("usage") or {}
        return {
            "content": message.get("content"),
            "tool_calls": calls,
            "usage": {
                "input_tokens": int(usage.get("prompt_tokens", 0)),
                "output_tokens": int(usage.get("completion_tokens", 0)),
            },
        }
