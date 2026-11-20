# Changelog

## 0.1.0

- Event-sourced SQLite journal with leases, fencing and unique-prefix run lookup.
- Deterministic replay of steps, model calls, tool calls, timers and signals.
- Retries with deterministic jitter; token, cost, call and step budgets.
- Durable tool-calling agent loop with human approval gates.
- Forking from any event; OpenTelemetry (OTLP/JSON) export; `revenant` CLI.
- `ScriptedModel` and a dependency-free OpenAI-compatible client.
