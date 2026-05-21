import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from healthcare_rag.eval_metrics import recall_at_k, reciprocal_rank
from healthcare_rag.retrieval import retrieve

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DOCS_PATH = ROOT / "docs"

# Run with uv run eval/metrics.py

def main():
    gt_path = ROOT / "eval" / "ground_truth.json"
    with open(gt_path) as f:
        ground_truth = json.load(f)

    doc_questions = [
        q for q in ground_truth
        if q.get("is_in_corpus") and q.get("expected_tool") == "vector_search"
        and q.get("expected_page") is not None
    ]

    if not doc_questions:
        print("No scoreable doc questions found.")
        return

    print(f"Scoring {len(doc_questions)} document questions.\n")

    recalls = {1: [], 3: [], 5: []}
    rrs = []
    rows = []

    for q in doc_questions:
        chunks = retrieve(q["question"], top_k=5)
        pages = [c.page_number for c in chunks]

        r1 = recall_at_k(pages, q["expected_page"], 1)
        r3 = recall_at_k(pages, q["expected_page"], 3)
        r5 = recall_at_k(pages, q["expected_page"], 5)
        rr = reciprocal_rank(pages, q["expected_page"])

        recalls[1].append(r1)
        recalls[3].append(r3)
        recalls[5].append(r5)
        rrs.append(rr)

        rows.append({
            "id": q["id"],
            "question": q["question"][:60],
            "expected_page": q["expected_page"],
            "retrieved_pages": pages,
            "r@1": r1, "r@3": r3, "r@5": r5, "rr": round(rr, 3),
        })

    print(f"{'ID':<6} {'R@1':>4} {'R@3':>4} {'R@5':>4} {'RR':>5}  Question")
    print("-" * 70)
    for r in rows:
        print(
            f"{r['id']:<6} {r['r@1']:>4} {r['r@3']:>4} {r['r@5']:>4} "
            f"{r['rr']:>5}  {r['question']}"
        )

    results = {
        "recall@1": float(np.mean(recalls[1])),
        "recall@3": float(np.mean(recalls[3])),
        "recall@5": float(np.mean(recalls[5])),
        "mrr":      float(np.mean(rrs)),
    }

    print("\nBaseline retrieval metrics: ")
    for k, v in results.items():
        print(f"  {k:12s}: {v:.3f}")

    DOCS_PATH.mkdir(exist_ok=True)
    md = (
        "# Retrieval Metrics\n\n"
        f"Scored on {len(doc_questions)} hand-written in-corpus document "
        "questions from `eval/ground_truth.json`.\n\n"
        "| Metric    | Value |\n"
        "|-----------|------:|\n"
        f"| Recall@1  | {results['recall@1']:.3f} |\n"
        f"| Recall@3  | {results['recall@3']:.3f} |\n"
        f"| Recall@5  | {results['recall@5']:.3f} |\n"
        f"| MRR       | {results['mrr']:.3f} |\n"
    )
    (DOCS_PATH / "baseline.md").write_text(md)
    print("\nSaved to root/docs/baseline.md")

if __name__ == "__main__":
    main()