import json
import sys
import time
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

from sqlalchemy import text

from healthcare_rag.core import get_engine
from healthcare_rag.retrieval import retrieve

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = [("structure", "hybrid_chunker"), ("fixed", "fixed")]

# Ablation 3: Does structure-aware chunking outperform fixed-size 512-token chunking?

# Pre-requisite: uv run root/scripts/ingest_fixed_chunks.py
# Run: root/eval/ablations/run_all.py to build docs/ablations.md.

def _page_set(expected_page) -> set[int]:
    if expected_page is None:
        return set()
    if isinstance(expected_page, list):
        return {int(p) for p in expected_page if p is not None}
    return {int(expected_page)}


def _fixed_chunk_count() -> int:
    with get_engine().connect() as conn:
        return conn.execute(
            text("SELECT count(*) FROM chunks WHERE chunk_strategy = 'fixed'")
        ).scalar() or 0


def run_ablation(ground_truth: list[dict]) -> dict:
    count = _fixed_chunk_count()
    if count == 0:
        raise RuntimeError(
            "No fixed-size chunks in the database. "
            "Run: uv run scripts/ingest_fixed_chunks.py"
        )
    print(f"Found {count} fixed-size chunks in database.")

    doc_qs = [
        q for q in ground_truth
        if q.get("is_in_corpus")
        and q.get("expected_tool") == "vector_search"
        and q.get("expected_page") is not None
    ]
    if not doc_qs:
        raise ValueError("No scoreable doc questions found.")

    print(f"Running chunking ablation on {len(doc_qs)} document questions")

    results = {}
    for label, strategy in CONDITIONS:
        print(f"\nStrategy: {label} (chunk_strategy='{strategy}')")
        rows = []
        t0 = time.time()
        # Reranker OFF to isolate the chunking effect (Ablation 2 handles reranker).
        for q in doc_qs:
            expected = _page_set(q.get("expected_page"))
            chunks = retrieve(
                q["question"], top_k=10, mode="hybrid",
                do_rerank=False, strategy=strategy,
            )
            pages = [c.page_number for c in chunks]

            def hit(k: int) -> int:
                return int(bool(expected & set(pages[:k])))

            rr = next((1.0 / (i + 1) for i, p in enumerate(pages) if p in expected), 0.0)

            rows.append({"id": q["id"], "hits": {k: hit(k) for k in (1, 3, 5, 10)}, "rr": rr})
            print(f"{q['id']} hit@5={hit(5)} rr={rr:.3f}")

        elapsed = time.time() - t0
        agg = {
            f"recall@{k}": round(float(np.mean([r["hits"][k] for r in rows])), 3)
            for k in (1, 3, 5, 10)
        }
        agg["mrr"] = round(float(np.mean([r["rr"] for r in rows])), 3)
        print(f"-> Recall@5={agg['recall@5']:.3f} MRR={agg['mrr']:.3f} ({elapsed:.1f}s)")

        results[label] = {"aggregates": agg, "per_question": rows, "strategy": strategy}

    return results


def section_md(results: dict) -> str:
    s = results["structure"]["aggregates"]
    fx = results["fixed"]["aggregates"]
    dr5 = s["recall@5"] - fx["recall@5"]
    dmrr = s["mrr"] - fx["mrr"]

    table = (
        "| Strategy        | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR   |\n"
        "|-----------------|----------|----------|----------|-----------|-------|\n"
        f"| structure-aware | {s['recall@1']:.3f}    | {s['recall@3']:.3f}    | {s['recall@5']:.3f}    | {s['recall@10']:.3f}     | {s['mrr']:.3f} |\n"
        f"| fixed-size 512  | {fx['recall@1']:.3f}    | {fx['recall@3']:.3f}    | {fx['recall@5']:.3f}    | {fx['recall@10']:.3f}     | {fx['mrr']:.3f} |"
    )

    if dr5 > 0:
        interp = (
            f"Structure-aware chunking outperforms fixed-size on Recall@5 by {dr5:+.3f} "
            f"and MRR by {dmrr:+.3f}. The CMS Final Rule has deep section nesting — "
            f"regulatory subsections are coherent units of meaning, and the Docling "
            f"HybridChunker preserves those boundaries. Fixed-size 512-token chunks cut "
            f"across section boundaries, fragmenting regulatory requirements and diluting "
            f"the information density per chunk."
        )
    else:
        interp = (
            f"Fixed-size chunking outperforms structure-aware on Recall@5 by "
            f"{abs(dr5):+.3f} and MRR by {abs(dmrr):+.3f}. The structure-aware chunks "
            f"were built with CHUNK_MAX_TOKENS=400, producing smaller chunks than the "
            f"512-token fixed windows. On this corpus the larger windows capture more "
            f"relevant context per chunk. Structure-aware is still used in production "
            f"for its section_path metadata and cleaner citations."
        )

    return f"""\
## Ablation 3 — Chunking strategy: structure-aware vs fixed-size

**Question:** Does structure-aware (Docling HybridChunker) chunking outperform naive
fixed-size 512-token chunking with 50-token overlap?

**Setup:** hybrid mode, `do_rerank=False`, `top_k=10` in both conditions.

{table}

**Interpretation:** {interp}
"""

# Entry point — writes JSON only. Run run_all.py to build docs/ablations.md.
def main() -> None:
    print("Error: run root/eval/ablations/run_all.py to build docs/ablations.md")

if __name__ == "__main__":
    main()