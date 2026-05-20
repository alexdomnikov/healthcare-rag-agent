import asyncio
import json
import re
from pathlib import Path

import instructor
import numpy as np
from dotenv import load_dotenv
from ragas.llms.base import InstructorLLM
from ragas.metrics.collections.context_precision import ContextPrecision
from ragas.metrics.collections.context_recall import ContextRecall
from ragas.metrics.collections.faithfulness import Faithfulness

load_dotenv()

JUDGE_MODEL = "llama-3.3-70b-versatile"
RESPONSES_PATH = Path("ground_truth_responses_v1.json")
PLACEHOLDER = "[No chunks retrieved — misrouted or tool error.]"

# Ragas + custom metrics eval
# Run: uv run eval/llm_eval.py

# Ragas
def build_metrics() -> dict:
    llm = InstructorLLM(
        client=instructor.from_provider(f"groq/{JUDGE_MODEL}", async_client=True),
        model=JUDGE_MODEL,
        provider="groq",
    )
    return {
        "faithfulness": Faithfulness(llm=llm),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }

async def score_question(r: dict, metrics: dict) -> dict:
    chunks = r.get("retrieved_chunks") or [PLACEHOLDER]
    try:
        faith = await metrics["faithfulness"].ascore(
            user_input=r["question"],
            response=r["final_answer"],
            retrieved_contexts=chunks,
        )
        ctx_p = await metrics["context_precision"].ascore(
            user_input=r["question"],
            reference=r["expected_answer"],
            retrieved_contexts=chunks,
        )
        ctx_r = await metrics["context_recall"].ascore(
            user_input=r["question"],
            retrieved_contexts=chunks,
            reference=r["expected_answer"],
        )
        return {
            "faithfulness": round(float(faith), 3),
            "context_precision": round(float(ctx_p), 3),
            "context_recall": round(float(ctx_r), 3),
        }
    except Exception as e:
        print(f"scoring failed: {e}")
        return {"faithfulness": None, "context_precision": None, "context_recall": None}

async def run_ragas(doc_qs: list[dict], metrics: dict) -> list[dict]:
    results = []
    for i, r in enumerate(doc_qs):
        print(f" [{i+1}/{len(doc_qs)}] {r['id']}...")
        results.append(await score_question(r, metrics))
        await asyncio.sleep(4)
    return results

# Deterministic metrics
def to_page_set(expected_page) -> set[int]:
    if expected_page is None: return set()
    if isinstance(expected_page, list): return set(expected_page)
    return {expected_page}

def citation_f1(answer: str, expected_page) -> float | None:
    cited = {int(p) for p in re.findall(r"\[p\.\s*(\d+)\]", answer)}
    expected = to_page_set(expected_page)
    if not expected or not cited:
        return 0.0 if expected else None
    p = len(cited & expected) / len(cited)
    r = len(cited & expected) / len(expected)
    return round(2 * p * r / (p + r), 3) if (p + r) else 0.0

def routing_accuracy(responses: list[dict]) -> dict:
    misrouted = [r for r in responses if r["called_tool"] != r["expected_tool"]]
    return {
        "accuracy": round(1 - len(misrouted) / len(responses), 3),
        "correct": len(responses) - len(misrouted),
        "total": len(responses),
        "misrouted": [{"id": r["id"], "expected": r["expected_tool"], "called": r["called_tool"]}
                      for r in misrouted],
    }

def hallucination_rate(responses: list[dict]) -> float | None:
    oos = [r for r in responses if not r["is_in_corpus"] and r["expected_tool"] == "none"]
    if not oos: return None
    refusals = ["don't have", "do not have", "don't know", "cannot answer"]
    fabricated = [r for r in oos if not any(p in r["final_answer"].lower() for p in refusals)]
    return round(len(fabricated) / len(oos), 3)

# Output
def build_markdown(summary: dict, per_q: list[dict], routing: dict) -> str:
    f = lambda v: f"**{v:.3f}**" if v is not None else "—"
    g = lambda v: f"{v:.3f}" if v is not None else "—"
    lines = [
        "# Baseline Evaluation",
        f"\n> Judge: `{JUDGE_MODEL}` via Groq\n",
        "## Summary\n",
        "| Metric | Value |", "|--------|-------|",
        f"| Tool routing accuracy    | {f(summary['routing_accuracy'])} ({routing['correct']}/{routing['total']}) |",
        f"| Hallucination rate (OOS) | {f(summary['hallucination_rate'])} |",
        f"| Citation F1 (avg)        | {f(summary['citation_f1_avg'])} |",
        f"| Faithfulness             | {f(summary['faithfulness'])} |",
        f"| Context precision        | {f(summary['context_precision'])} |",
        f"| Context recall           | {f(summary['context_recall'])} |",
    ]
    if routing["misrouted"]:
        lines += ["\n## Misrouted\n", "| ID | Expected | Called |", "|----|----------|--------|"]
        for m in routing["misrouted"]:
            lines.append(f"| {m['id']} | {m['expected']} | {m['called']} |")
    lines += [
        "\n## Per-Question\n",
        "| ID | Faith | Ctx.P | Ctx.R | CiteF1 | Routed |",
        "|----|-------|-------|-------|--------|--------|",
    ]
    for q in per_q:
        ok = "+" if q["routing_correct"] else "x"
        lines.append(f"| {q['id']} | {g(q['faithfulness'])} | {g(q['context_precision'])} "
                     f"| {g(q['context_recall'])} | {g(q['citation_f1'])} | {ok} |")
    return "\n".join(lines) + "\n"

# Main
def main():
    responses = json.loads(RESPONSES_PATH.read_text())
    doc_qs = [r for r in responses if r["is_in_corpus"] and r["expected_tool"] == "vector_search"]

    routing = routing_accuracy(responses)
    hall = hallucination_rate(responses)
    for r in doc_qs:
        r["_cite_f1"] = citation_f1(r["final_answer"], r.get("expected_page"))
        r["_routing_correct"] = r["called_tool"] == r["expected_tool"]

    print(f"Routing accuracy : {routing['accuracy']:.3f}  ({routing['correct']}/{routing['total']})")
    for m in routing["misrouted"]:
        print(f"x {m['id']} expected={m['expected']} called={m['called']}")
    print(f"Hallucination OOS: {hall}")
    cite_vals = [r["_cite_f1"] for r in doc_qs if r["_cite_f1"] is not None]
    print(f"Citation F1 avg : {round(float(np.mean(cite_vals)), 3)}")

    print(f"\nBuilding Ragas metrics (judge: {JUDGE_MODEL})")
    metrics = build_metrics()

    print(f"Scoring {len(doc_qs)} questions (3 metrics each)\n")
    ragas_scores = asyncio.run(run_ragas(doc_qs, metrics))

    per_q = [
        {
            "id": r["id"],
            "faithfulness": s["faithfulness"],
            "context_precision": s["context_precision"],
            "context_recall": s["context_recall"],
            "citation_f1": r["_cite_f1"],
            "routing_correct": r["_routing_correct"],
        }
        for r, s in zip(doc_qs, ragas_scores)
    ]

    def avg(key):
        vals = [q[key] for q in per_q if q.get(key) is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    summary = {
        "routing_accuracy": routing["accuracy"],
        "hallucination_rate": hall,
        "citation_f1_avg": avg("citation_f1"),
        "faithfulness": avg("faithfulness"),
        "context_precision": avg("context_precision"),
        "context_recall": avg("context_recall"),
        "routing_detail": routing,
        "per_question": per_q,
    }

    print("\n-- Summary --")
    for k, v in summary.items():
        if not isinstance(v, (dict, list)):
            print(f"  {k:<28} {v}")

    Path("../docs").mkdir(exist_ok=True)
    Path("eval_baseline.json").write_text(json.dumps(summary, indent=2))
    Path("../docs/eval_baseline.md").write_text(build_markdown(summary, per_q, routing))
    print("\nSaved -> eval/eval_baseline.json  docs/eval_baseline.md")

if __name__ == "__main__":
    main()