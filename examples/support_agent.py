"""A support agent that must get human approval before issuing a refund.

Runs fully offline with a scripted model. Shows a run pausing for approval,
being approved from the "outside", and then a time-travel fork that replays the
same conversation but denies the refund instead.

    python -m examples.support_agent                      # the whole story in one process

Or drive it by hand from the CLI (run from the repository root):

    python -m examples.support_agent --start              # prints the run id, then pauses
    revenant -i examples.support_agent show <run>
    revenant -i examples.support_agent approve <run>
"""

from __future__ import annotations

import sys

from revenant import Budget, Runtime, ScriptedModel, run_agent, tool, workflow

ORDERS = {"A-1042": {"customer": "dana@example.com", "total": 89.90, "status": "delivered"}}
REFUNDS: dict[str, dict] = {}


@tool
def lookup_order(order_id: str) -> dict:
    """Fetch an order by id."""
    return ORDERS.get(order_id, {"error": "order not found"})


@tool(requires_approval=True)
def issue_refund(order_id: str, amount: float, idempotency_key: str = "") -> dict:
    """Refund money to the customer. Requires human approval."""
    REFUNDS.setdefault(idempotency_key, {"order_id": order_id, "amount": amount})
    return {"refunded": amount, "order_id": order_id}


def policy(messages: list[dict], tools: list[dict]) -> dict | str:
    """Stand-in for an LLM: decides the next move from the conversation so far."""
    last = messages[-1]
    if last["role"] == "user":
        return {
            "content": "Let me look up that order.",
            "tool_calls": [{"name": "lookup_order", "arguments": {"order_id": "A-1042"}}],
        }
    if last["name"] == "lookup_order":
        return {
            "content": "The order was delivered damaged; a full refund is warranted.",
            "tool_calls": [{"name": "issue_refund", "arguments": {"order_id": "A-1042", "amount": 89.90}}],
        }
    if "denied" in last["content"]:
        return "I could not issue the refund: a reviewer declined it. I have escalated your case."
    return "Your refund of $89.90 for order A-1042 has been issued."


MODEL = ScriptedModel(policy, name="scripted-support-model", input_price=3.0, output_price=15.0)


@workflow(name="examples.support_agent")
def support_agent(ctx, ticket):
    return run_agent(
        ctx,
        MODEL,
        [lookup_order, issue_refund],
        ticket,
        system="You are a support agent. Refunds need human approval.",
    )


def main() -> None:
    db = "support_demo.db"
    ticket = "Order A-1042 arrived broken. I want my money back."
    with Runtime(db) as rt:
        paused = rt.start(support_agent, ticket, budget=Budget(max_cost_usd=0.05, max_llm_calls=8))
        if "--start" in sys.argv:
            print(paused.run_id)
            return

        print(f"run {paused.run_id[:12]} is {paused.status}")
        print(f"  waiting on: {rt.pending(paused.run_id)}")
        print(f"  refunds so far: {len(REFUNDS)}  (nothing risky runs before approval)\n")

        done = rt.approve(paused.run_id, reason="photos confirm damage")
        print(f"approved  -> {done.result['output']}")
        print(f"  refunds: {list(REFUNDS.values())}")
        print(f"  usage: {done.usage}\n")

        # Time travel: branch from just before the human decision and deny it.
        events = rt.journal.events(paused.run_id)
        gate = next(e for e in events if e.type == "signal_waiting")
        fork_id = rt.fork(paused.run_id, at_seq=gate.seq + 1)
        rt.resume(fork_id)  # replays both model calls from the journal: zero new tokens
        denied = rt.approve(fork_id, approved=False, reason="outside refund window")
        print(f"fork {fork_id[:12]} denied -> {denied.result['output']}")
        print(f"  refunds still: {len(REFUNDS)}\n")
        print(f"Inspect:  revenant --db {db} runs")
        print(f"          revenant --db {db} show {paused.run_id[:12]}")


if __name__ == "__main__":
    main()
