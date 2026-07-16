import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

# Runs every question in ground_truth.json through the agent and records which
# tool was called. Misrouted questions land in eval/misrouted.json as the
# iteration target for prompt tuning.
#
# Usage:
#   uv run python eval/routing_eval.py
#   uv run python eval/routing_eval.py --gt eval/ground_truth.json --verbose


def extract_first_tool_called(messages: list[Any]) -> str:
    # AIMessage.tool_calls is the model's declared tool call; ToolMessage.name
    # is the fallback for the rare case where tool_calls isn't populated.
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            return msg.tool_calls[0]["name"]
    for msg in messages:
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None):
            return msg.name
    return "none"


def run_routing_eval(
    ground_truth_path: str = str(ROOT / "eval" / "ground_truth.json"),
    output_path: str = str(ROOT / "eval" / "misrouted.json"),
    verbose: bool = True,
    workers: int = 1,
) -> tuple[float, list[dict]]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from healthcare_rag.core import get_agent, get_embed_model, get_reranker  # type: ignore[import]

    # Pre-warm singletons; lru_cache isn't thread-safe on first call, and without
    # this the worker threads race to load the same models.
    agent = get_agent()
    get_embed_model()
    get_reranker()

    gt_path = Path(ground_truth_path)
    if not gt_path.exists():
        print(f"ERROR: ground truth file not found: {gt_path}", file=sys.stderr)
        sys.exit(1)

    with open(gt_path) as f:
        ground_truth: list[dict] = json.load(f)

    col_w = 20
    if verbose:
        print(f"\nRouting eval: {len(ground_truth)} questions\n")
        print(f"{'ID':<8} {'Expected':<{col_w}} {'Called':<{col_w}} {'Latency':>8}")
        print("─" * (8 + col_w * 2 + 14))

    def _eval_one(item: dict) -> dict:
        expected = item.get("expected_tool", "none")
        t0 = time.perf_counter()
        try:
            result = agent.invoke({"messages": [HumanMessage(content=item["question"])]})
            called = extract_first_tool_called(result["messages"])
            final_answer = result["messages"][-1].content
        except Exception as exc:
            called = "error"
            final_answer = f"Exception: {exc}"
        return {
            "id": item.get("id", "?"),
            "question": item["question"],
            "expected_tool": expected,
            "called_tool": called,
            "correct": called == expected,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "is_in_corpus": item.get("is_in_corpus"),
            "final_answer_preview": final_answer[:120],
        }

    all_results: list[dict] = []
    misrouted: list[dict] = []
    correct = 0

    # Groq free tier (8K TPM on gpt-oss-120b with retrieval context) overruns
    # above 1-2 concurrent requests. Set --workers higher once on a paid tier.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_eval_one, item): item for item in ground_truth}
        for future in as_completed(futures):
            r = future.result()
            all_results.append(r)
            if r["called_tool"] == "error":
                symbol = "e"
            elif r["correct"]:
                correct += 1
                symbol = "+"
            else:
                symbol = "x"
                misrouted.append({
                    "id": r["id"],
                    "question": r["question"],
                    "expected_tool": r["expected_tool"],
                    "called_tool": r["called_tool"],
                    "is_in_corpus": r["is_in_corpus"],
                    "final_answer_preview": r["final_answer_preview"],
                })
            if verbose:
                print(
                    f"{str(r['id']):<8} {r['expected_tool']:<{col_w}} "
                    f"{r['called_tool']:<{col_w}} {r['latency_ms']:>7}ms {symbol}"
                )

    # Exclude rate-limit errors from accuracy so a 429 storm doesn't tank the score.
    errors = [r for r in all_results if r["called_tool"] == "error"]
    valid = [r for r in all_results if r["called_tool"] != "error"]
    accuracy = sum(r["correct"] for r in valid) / len(valid) if valid else 0.0
    
    if verbose:
        print("─" * (8 + col_w * 2 + 14))
        print(f"\nTool-routing accuracy: {sum(r['correct'] for r in valid)}/{len(valid)} = {accuracy:.1%}"
           + (f" ({len(errors)} excluded due to rate limit errors)" if errors else ""))
        if misrouted:
            print(f"\nMisrouted questions ({len(misrouted)}):")
            for m in misrouted:
                print(
                    f"  [{m['id']}] expected={m['expected_tool']!r:12} "
                    f"got={m['called_tool']!r:18} — {m['question'][:60]}"
                )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(
            {"accuracy": accuracy, "correct": correct,
             "total": len(ground_truth), "misrouted": misrouted},
            f, indent=2,
        )

    full_path = out_path.parent / "routing_eval_full.json"
    with open(full_path, "w") as f:
        json.dump(all_results, f, indent=2)

    if verbose:
        print(f"\nMisrouted questions -> {out_path}")
        print(f"Full per-question log -> {full_path}\n")

    return accuracy, misrouted

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the tool-routing evaluation for the healthcare RAG agent."
    )
    p.add_argument(
        "--gt",
        default=str(ROOT / "eval" / "ground_truth.json"),
        metavar="PATH",
        help="Path to ground_truth.json  (default: eval/ground_truth.json)",
    )
    p.add_argument(
        "--output",
        default=str(ROOT / "eval" / "misrouted.json"),
        metavar="PATH",
        help="Where to save misrouted.json  (default: eval/misrouted.json)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        metavar="FLOAT",
        help="Accuracy threshold for exit code 0  (default: 0.85)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Concurrent agent invocations. Free Groq tier handles ~1; "
             "bump on paid tiers (default: 1)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-question output",
    )
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    accuracy, misrouted = run_routing_eval(
        ground_truth_path=args.gt,
        output_path=args.output,
        verbose=not args.quiet,
        workers=args.workers,
    )

    threshold = args.threshold
    if accuracy >= threshold:
        print(f"Success! {accuracy:.1%} >= {threshold:.0%} target. Agent ready for finishing touches.")
        sys.exit(0)
    else:
        print(
            f"Failure: {accuracy:.1%} < {threshold:.0%} target. review eval/misrouted.json and iterate on tool docstrings / system prompt."
        )
        sys.exit(1)