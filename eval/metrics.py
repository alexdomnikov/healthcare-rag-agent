import json
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from pathlib import Path

from healthcare_rag.retrieval import retrieve

load_dotenv()

# project root
ROOT = Path(__file__).resolve().parents[1]
DOCS_PATH = ROOT / 'docs'

# Calculates retrieval metrics.
# Run with uv run eval/metrics.py

def _to_page_set(expected_page) -> set[int]:
    # Normalize expected_page to a set whether it's an int, list, or None.
    if expected_page is None:
        return set()
    if isinstance(expected_page, list):
        return set(expected_page)
    return {expected_page}

def recall_at_k(retrieved_pages: list[int], expected_page, k: int) -> int:
    # Returns 1 if document page matches expected page, 0 if not.
    pages = _to_page_set(expected_page)
    if not pages:
        return 0
    return int(bool(pages & set(retrieved_pages[:k])))

def reciprocal_rank(retrieved_pages: list[int], expected_page) -> float:
    # Returns 1/(rank+1) of the first chunk that matches expected_page, or 0.0
    pages = _to_page_set(expected_page)
    if not pages:
        return 0.0
    for i, page in enumerate(retrieved_pages):
        if page in pages:
            return 1.0 / (i + 1)
    return 0.0

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
    md = f"""# Baseline Retrieval Metrics from Week 1:
        Recall@1: {results['recall@1']:.3f}
        Recall@3: {results['recall@3']:.3f}
        Recall@5: {results['recall@5']:.3f}
        MRR: {results['mrr']:.3f}

        Scored on {len(doc_questions)} handwritten document questions.
    """
    (DOCS_PATH / "baseline.md").write_text(md)
    print("\nSaved to root/docs/baseline.md")

if __name__ == "__main__":
    main()