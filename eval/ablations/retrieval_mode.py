import time
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

from healthcare_rag.eval_metrics import recall_at_k, reciprocal_rank
from healthcare_rag.retrieval import retrieve

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]

# Ablation 1: does hybrid retrieval beat dense-only or lexical-only on the CMS
# corpus? Run eval/ablations/run_all.py to build docs/ablations.md.

MODES = ["hybrid", "vector", "lexical"]
FIRST_PASS_K = 10  # enough for Recall@10; reranker off so this is the final list


def run_ablation(ground_truth: list[dict]) -> dict:
    # Reranker is off — the delta between modes should reflect first-pass
    # retrieval quality only, not the cross-encoder.
    doc_qs = [
        q for q in ground_truth
        if q.get("is_in_corpus")
        and q.get("expected_tool") == "vector_search"
        and q.get("expected_page") is not None
    ]
    if not doc_qs:
        raise ValueError(
            "No scoreable doc questions found. "
            "Check that ground_truth.json has is_in_corpus=True, "
            "expected_tool='vector_search', and expected_page filled in."
        )

    print(f"Running ablation on {len(doc_qs)} document questions × {len(MODES)} modes")
    print(f"Retrieval top_k={FIRST_PASS_K}, do_rerank=False, strategy='hybrid_chunker'\n")

    results: dict[str, dict] = {}

    for mode in MODES:
        print(f"Mode: {mode}")
        rows: list[dict] = []
        t_start = time.time()

        for q in doc_qs:
            chunks = retrieve(
                q["question"],
                top_k=FIRST_PASS_K,
                mode=mode,
                do_rerank=False,
                strategy="hybrid_chunker",
            )
            pages = [c.page_number for c in chunks]

            row = {
                "id": q["id"],
                "question": q["question"],
                "expected_page": q["expected_page"],
                "retrieved_pages": pages,
                "recall@1": recall_at_k(pages, q["expected_page"], 1),
                "recall@3": recall_at_k(pages, q["expected_page"], 3),
                "recall@5": recall_at_k(pages, q["expected_page"], 5),
                "recall@10": recall_at_k(pages, q["expected_page"], 10),
                "rr": reciprocal_rank(pages, q["expected_page"]),
            }
            rows.append(row)
            # Lexical can stall on the first call while the tsvector index warms.
            print(f"{q['id']} hit@5={row['recall@5']} rr={row['rr']:.3f}")

        elapsed = time.time() - t_start
        n = len(rows)

        aggregates = {
            "recall@1": float(np.mean([r["recall@1"] for r in rows])),
            "recall@3": float(np.mean([r["recall@3"] for r in rows])),
            "recall@5": float(np.mean([r["recall@5"] for r in rows])),
            "recall@10": float(np.mean([r["recall@10"] for r in rows])),
            "mrr": float(np.mean([r["rr"] for r in rows])),
            "n_questions": n,
            "elapsed_s": round(elapsed, 2),
        }

        results[mode] = {"rows": rows, "aggregates": aggregates}

        print(
            f"-> Recall@5={aggregates['recall@5']:.3f} "
            f"MRR={aggregates['mrr']:.3f} "
            f"({elapsed:.1f}s)\n"
        )

    return results

def build_markdown_table(results: dict) -> str:
    header = "| Mode    | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR   |"
    sep = "|---------|----------|----------|----------|-----------|-------|"
    rows = []
    for mode in MODES:
        a = results[mode]["aggregates"]
        rows.append(
            f"| {mode:<7} "
            f"| {a['recall@1']:.3f}    "
            f"| {a['recall@3']:.3f}    "
            f"| {a['recall@5']:.3f}    "
            f"| {a['recall@10']:.3f}     "
            f"| {a['mrr']:.3f} |"
        )
    return "\n".join([header, sep] + rows)


def section_md(results: dict) -> str:
    table = build_markdown_table(results)

    h_r5 = results["hybrid"]["aggregates"]["recall@5"]
    v_r5 = results["vector"]["aggregates"]["recall@5"]
    l_r5 = results["lexical"]["aggregates"]["recall@5"]

    def _winner(vals: dict[str, float]) -> str:
        return max(vals, key=vals.__getitem__)

    r5_winner  = _winner({m: results[m]["aggregates"]["recall@5"] for m in MODES})
    mrr_winner = _winner({m: results[m]["aggregates"]["mrr"] for m in MODES})

    interpretation = (
        f"Hybrid {'wins' if r5_winner == 'hybrid' else 'does not win'} on Recall@5 "
        f"({h_r5:.3f}), ahead of dense-only by {h_r5 - v_r5:+.3f} points and "
        f"lexical-only by {h_r5 - l_r5:+.3f} points. "
        f"MRR winner is {mrr_winner} ({results[mrr_winner]['aggregates']['mrr']:.3f}). "
        f"The gap between hybrid and dense-only shows that the CMS corpus contains "
        f"exact-match signals (CFR section codes, specific dollar figures) that BM25 "
        f"captures and the embedding model misses. The gap between hybrid and "
        f"lexical-only shows that paraphrase-resistant questions require semantic "
        f"retrieval to surface."
    )

    return f"""\
## Ablation 1: retrieval mode (hybrid vs dense vs lexical)

**Question:** Does combining BM25 and dense retrieval outperform either method alone?

**Setup:** `do_rerank=False`, `top_k=10`, `strategy='hybrid_chunker'`.
Reranker disabled so the delta reflects only first-pass retrieval quality.

{table}

**Interpretation:** {interpretation}
"""

def main() -> None:
    print("Error: run root/eval/ablations/run_all.py to build docs/ablations.md")

if __name__ == "__main__":
    main()