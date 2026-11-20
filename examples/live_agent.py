"""A durable agent on a real model, via any OpenAI-compatible endpoint.

    export OPENAI_API_KEY=...            # or point at a local server:
    export OPENAI_BASE_URL=http://localhost:11434/v1
    export REVENANT_MODEL=gpt-4o-mini
    python -m examples.live_agent "What is 17% of 2,340, plus the number of days in a leap year?"

Interrupt it with Ctrl-C at any point and run the same command with
`--resume <run id>`: completed model and tool calls are replayed from the
journal instead of being paid for again.
"""

from __future__ import annotations

import ast
import operator
import os
import sys

from revenant import Budget, OpenAICompatibleModel, Runtime, run_agent, tool, workflow

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_evaluate(node.operand))
    raise ValueError("only arithmetic on numbers is supported")


@tool
def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression such as '0.17 * 2340 + 366'."""
    return _evaluate(ast.parse(expression, mode="eval").body)


@workflow(name="examples.live_agent")
def live_agent(ctx, question):
    model = OpenAICompatibleModel(os.environ.get("REVENANT_MODEL", "gpt-4o-mini"))
    return run_agent(
        ctx,
        model,
        [calculate],
        question,
        system="Use the calculate tool for every piece of arithmetic. Answer concisely.",
    )


def main() -> None:
    with Runtime("live_agent.db") as rt:
        if len(sys.argv) == 3 and sys.argv[1] == "--resume":
            result = rt.resume(sys.argv[2], force=True)
        elif len(sys.argv) == 2:
            result = rt.start(live_agent, sys.argv[1], budget=Budget(max_llm_calls=10))
        else:
            raise SystemExit(__doc__)
        print(f"run {result.run_id}: {result.status}")
        print(result.result["output"] if result.completed else result.error)
        print(f"usage: {result.usage}")


if __name__ == "__main__":
    main()
