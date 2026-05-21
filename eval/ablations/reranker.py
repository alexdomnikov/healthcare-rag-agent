import json
import sys
import time
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

from healthcare_rag.retrieval import retrieve

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
STRATEGY = "hybrid_chunker"
CONDITIONS = [("off", False), ("on", True)]

# Ablation 2: Does cross-encoder reranking improve Recall@5 over raw hybrid retrieval?

# Run: root/eval/ablations/run_all.py to build docs/ablations.md.

def _page_set(expected_page) -> set[int]:
    if expected_page is None:
        return set()
    if isinstance(expected_page, list):
        return {int(p) for p in expected_page if p is not None}
    return {int(expected_page)}

def run_ablation(ground_truth: list[dict]) -> dict:
    doc_qs = [
        q for q in ground_truth
        if q.get("is_in_corpus")
        and q.get("expected_tool") == "vector_search"
        and q.get("expected_page") is not None
    ]
    if not doc_qs:
        raise ValueError("No scoreable doc questions found.")

    print(f"Running reranker ablation on {len(doc_qs)} document questions")
    print("(reranker-on fetches 50 candidates first. First run downloads weights)")

    results = {}
    for label, do_rerank in CONDITIONS:
        print(f"\nReranker: {label}")
        rows = []
        t0 = time.time()

        for q in doc_qs:
            expected = _page_set(q.get("expected_page"))
            chunks = retrieve(
                q["question"], top_k=5, mode="hybrid",
                do_rerank=do_rerank, strategy=STRATEGY,
            )
            pages = [c.page_number for c in chunks]

            def hit(k: int) -> int:
                return int(bool(expected & set(pages[:k])))

            rr = next((1.0 / (i + 1) for i, p in enumerate(pages) if p in expected), 0.0)

            rows.append({"id": q["id"], "hits": {k: hit(k) for k in (1, 3, 5)}, "rr": rr})
            print(f"{q['id']} hit@5={hit(5)} rr={rr:.3f}")

        elapsed = time.time() - t0
        agg = {f"recall@{k}": round(float(np.mean([r["hits"][k] for r in rows])), 3) for k in (1, 3, 5)}
        agg["mrr"] = round(float(np.mean([r["rr"] for r in rows])), 3)
        print(f"-> Recall@5={agg['recall@5']:.3f} MRR={agg['mrr']:.3f} ({elapsed:.1f}s)")

        results[label] = {"aggregates": agg, "per_question": rows}

    return results


def section_md(results: dict) -> str:
    off = results["off"]["aggregates"]
    on = results["on"]["aggregates"]
    dr5 = on["recall@5"] - off["recall@5"]
    dmrr = on["mrr"] - off["mrr"]

    table = (
        "| Condition    | Recall@1 | Recall@3 | Recall@5 | MRR   |\n"
        "|--------------|----------|----------|----------|-------|\n"
        f"| reranker off | {off['recall@1']:.3f}    | {off['recall@3']:.3f}    | {off['recall@5']:.3f}    | {off['mrr']:.3f} |\n"
        f"| reranker on  | {on['recall@1']:.3f}    | {on['recall@3']:.3f}    | {on['recall@5']:.3f}    | {on['mrr']:.3f} |"
    )

    if dr5 > 0:
        interp = (
            f"Reranking improves Recall@5 by {dr5:+.3f} and MRR by {dmrr:+.3f}. "
            f"The cross-encoder rescores 50 first-pass candidates rather than just the "
            f"top-5, surfacing chunks that hybrid ranked between 6th and 50th. The hybrid "
            f"stage is strong at recall but mediocre at precision; the cross-encoder corrects that."
        )
    else:
        interp = (
            f"Reranking produces a {dr5:+.3f} change on Recall@5 and {dmrr:+.3f} on MRR. "
            f"The hybrid first-pass is already surfacing the correct chunk near the top, "
            f"leaving little room for the cross-encoder to improve."
        )

    return f"""\
## Ablation 2 — Reranker: on vs off

**Question:** Does cross-encoder reranking improve Recall@5 over raw hybrid retrieval?

**Setup:** hybrid mode, `strategy='hybrid_chunker'`, `top_k=5`. Reranker-on fetches
50 first-pass candidates then reranks; reranker-off returns the raw hybrid top 5.

{table}

**Interpretation:** {interp}
"""

def main() -> None:
    print("Error: run root/eval/ablations/run_all.py to build docs/ablations.md")

if __name__ == "__main__":
    main()