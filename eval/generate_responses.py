import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH = ROOT / "eval" / "ground_truth.json"
OUTPUT = ROOT / "eval" / "ground_truth_responses.json"

# Runs the live agent on every ground_truth question and records its outputs
# in the shape consumed by eval/llm_eval.py. Regenerate before locking in
# pre-deployment numbers so the LLM judge scores the CURRENT agent's answers,
# not a stale snapshot from an earlier config.
#
# Usage:
#   uv run python eval/generate_responses.py


def _first_tool_called(messages: list) -> str:
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            return m.tool_calls[0]["name"]
    for m in messages:
        if isinstance(m, ToolMessage) and getattr(m, "name", None):
            return m.name
    return "none"


def _first_tool_output(messages: list) -> list[str]:
    # Returns the first ToolMessage content split into chunks. For
    # vector_search this is the list of "[p. N | section] text" strings; for
    # sql/openfda it's a single-element list with the tool's output. Returns
    # [] when no tool was called (expected_tool='none' questions).
    for m in messages:
        if isinstance(m, ToolMessage):
            content = m.content if isinstance(m.content, str) else str(m.content)
            return [p for p in content.split("\n\n") if p.strip()]
    return []


def main() -> None:
    from healthcare_rag.core import get_agent, get_embed_model, get_reranker

    agent = get_agent()
    get_embed_model()
    get_reranker()

    ground_truth: list[dict] = json.loads(GROUND_TRUTH.read_text())
    print(f"\nGenerating responses for {len(ground_truth)} question(s)\n")

    responses: list[dict] = []
    for q in ground_truth:
        t0 = time.perf_counter()
        try:
            result = agent.invoke({"messages": [HumanMessage(content=q["question"])]})
            final_answer = result["messages"][-1].content
            called_tool = _first_tool_called(result["messages"])
            retrieved_chunks = _first_tool_output(result["messages"])
        except Exception as exc:
            final_answer = f"Exception: {type(exc).__name__}: {exc}"
            called_tool = "error"
            retrieved_chunks = []
        ms = int((time.perf_counter() - t0) * 1000)
        ok = "+" if called_tool == q.get("expected_tool", "none") else "x"
        print(f"  {q['id']:<6} {ok}  exp={q.get('expected_tool', 'none'):<14} got={called_tool:<22} {ms:>6}ms")

        responses.append({
            "id": q["id"],
            "question": q["question"],
            "final_answer": final_answer,
            "retrieved_chunks": retrieved_chunks,
            "called_tool": called_tool,
            "expected_tool": q.get("expected_tool", "none"),
            "expected_answer": q.get("expected_answer"),
            "expected_page": q.get("expected_page"),
            "is_in_corpus": q.get("is_in_corpus", False),
        })

    OUTPUT.write_text(json.dumps(responses, indent=2))
    print(f"\nWrote {len(responses)} responses -> {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
