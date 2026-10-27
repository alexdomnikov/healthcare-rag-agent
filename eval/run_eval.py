import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH = ROOT / "eval" / "ground_truth.json"

# CI smoke subset. Each ID is verified in the saved baselines:
#   - routed to its expected tool (see eval/routing_eval_full.json)
#   - if vector_search, hits Recall@5 (see eval/ground_truth_responses.json)
# Replace IDs only after confirming the same in the latest baseline run.
SMOKE_IDS = ["q001", "q002", "q006", "q019", "q024"]


def _first_tool(messages: list) -> str:
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            return m.tool_calls[0]["name"]
    for m in messages:
        if isinstance(m, ToolMessage) and getattr(m, "name", None):
            return m.name
    return "none"


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    raw = response.headers.get("retry-after") if response is not None else None
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="CI eval orchestrator. Runs retrieval Recall@5 and tool-routing "
                    "accuracy on a subset of ground_truth.json and exits non-zero "
                    "if either falls below its threshold."
    )
    ap.add_argument("--subset", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--threshold-recall5", type=float, default=0.7)
    ap.add_argument("--threshold-routing", type=float, default=0.8)
    ap.add_argument(
        "--routing-delay",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Pause between agent invocations. A doc question costs roughly 4-5K "
             "tokens once retrieval context is in the prompt, so back-to-back "
             "calls blow through Groq's free-tier 8K TPM and come back 429. "
             "Set to 0 on a paid tier (default: 30)",
    )
    args = ap.parse_args()

    gt: list[dict] = json.loads(GROUND_TRUTH.read_text())
    if args.subset == "smoke":
        gt = [q for q in gt if q["id"] in SMOKE_IDS]
        missing = set(SMOKE_IDS) - {q["id"] for q in gt}
        if missing:
            print(f"ERROR: smoke IDs missing from ground_truth.json: {sorted(missing)}", file=sys.stderr)
            sys.exit(2)

    # Imports are deferred so --help works without DB / model deps installed.
    import os

    from groq import RateLimitError
    from healthcare_rag.core import get_agent, get_embed_model, get_reranker
    from healthcare_rag.eval_metrics import recall_at_k
    from healthcare_rag.retrieval import retrieve

    agent = get_agent()
    get_embed_model()
    # Skip reranker warmup when bypassed (DISABLE_RERANKER=1 in CI). The
    # cross-encoder weights are ~570MB and the model is CPU-bound, so skipping
    # the download saves several minutes on a cold CI runner.
    if os.getenv("DISABLE_RERANKER") != "1":
        get_reranker()

    doc_qs = [
        q for q in gt
        if q.get("expected_tool") == "vector_search"
        and q.get("is_in_corpus")
        and q.get("expected_page") is not None
    ]
    print(f"\nRetrieval Recall@5 on {len(doc_qs)} doc question(s):")
    recalls: list[int] = []
    for q in doc_qs:
        chunks = retrieve(q["question"], top_k=5)
        pages = [c.page_number for c in chunks]
        r5 = recall_at_k(pages, q["expected_page"], 5)
        recalls.append(r5)
        print(f"  {q['id']}  r@5={r5}  retrieved={pages}  expected={q['expected_page']}")
    recall_5 = float(np.mean(recalls)) if recalls else 0.0

    def _route_once(question: str) -> str:
        res = agent.invoke({"messages": [HumanMessage(content=question)]})
        return _first_tool(res["messages"])

    print(f"\nTool routing on {len(gt)} question(s), {args.routing_delay:.0f}s apart:")
    routing_correct = 0
    for i, q in enumerate(gt):
        # Space the calls out rather than firing them back to back. A 429 counts
        # as a routing failure, which used to fail CI on quota rather than on
        # any real regression.
        if i and args.routing_delay:
            time.sleep(args.routing_delay)
        t0 = time.perf_counter()
        try:
            called = _route_once(q["question"])
        except RateLimitError as exc:
            # Groq's retry-after is authoritative; the delay is only a fallback.
            wait = _retry_after_seconds(exc) or args.routing_delay
            print(f"  {q['id']}  rate limited, retrying once in {wait:.0f}s")
            time.sleep(wait)
            try:
                called = _route_once(q["question"])
            except Exception as exc:
                called = f"error:{type(exc).__name__}"
        except Exception as exc:
            called = f"error:{type(exc).__name__}"
        ms = int((time.perf_counter() - t0) * 1000)
        ok = called == q.get("expected_tool", "none")
        routing_correct += int(ok)
        print(f"  {q['id']}  exp={q.get('expected_tool', 'none'):<14} got={called:<22} {ms:>6}ms  {'OK' if ok else 'FAIL'}")
    routing_acc = routing_correct / len(gt) if gt else 0.0

    print("\n--- Summary ---")
    print(f"  Recall@5          {recall_5:.3f}   threshold {args.threshold_recall5}")
    print(f"  Routing accuracy  {routing_acc:.3f}   threshold {args.threshold_routing}")

    failures: list[str] = []
    if doc_qs and recall_5 < args.threshold_recall5:
        failures.append(f"Recall@5 {recall_5:.3f} < {args.threshold_recall5}")
    if routing_acc < args.threshold_routing:
        failures.append(f"Routing {routing_acc:.3f} < {args.threshold_routing}")

    if failures:
        print("\nFAIL: " + "; ".join(failures))
        sys.exit(1)
    print("\nPASS")


if __name__ == "__main__":
    main()
