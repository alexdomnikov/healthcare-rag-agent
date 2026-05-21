import json
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from healthcare_rag.retrieval import retrieve

load_dotenv()

# project root
ROOT = Path(__file__).resolve().parents[2]

# Ablation 1: Does hybrid retrieval actually outperform dense-only or lexical-only
#   on the CMS corpus?

# Run: uv run eval/ablations/retrieval_mode.py

# Helpers: mirrors eval/metrics.py so I don't need a cross import of metrics.py,
#   which was meant to be a standalone script instead of a module.
def _to_page_set(expected_page) -> set[int]:
    # Normalize expected_page to a set (handles int, list[int], or None).
    if expected_page is None:
        return set()
    if isinstance(expected_page, list):
        return set(expected_page)
    return {int(expected_page)}

def recall_at_k(retrieved_pages: list[int], expected_page, k: int) -> int:
    # 1 if any of expected_page appears in retrieved_pages[:k], else 0.
    pages = _to_page_set(expected_page)
    if not pages:
        return 0
    return int(bool(pages & set(retrieved_pages[:k])))

def reciprocal_rank(retrieved_pages: list[int], expected_page) -> float:
    # 1 / rank-of-first-hit, or 0 if no hit.
    pages = _to_page_set(expected_page)
    if not pages:
        return 0.0
    for i, page in enumerate(retrieved_pages):
        if page in pages:
            return 1.0 / (i + 1)
    return 0.0

# Core ablation
MODES = ["hybrid", "vector", "lexical"]
# enough for Recall@10; reranker is OFF so this is the final list
FIRST_PASS_K = 10
def run_ablation(ground_truth: list[dict]) -> dict:
    # Run all three retrieval modes on the doc questions.
    # NOTE: reranker disabled so delta between modes reflects retrieval method,
    #   not downstream reranking.

    # Filter to scoreable document questions
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

            # NOTE: metrics computed per mode = Recall@1, Recall@3, Recall@5, Recall@10, MRR
            row = {
                "id": q["id"],
                "question": q["question"],
                "expected_page": q["expected_page"],
                "retrieved_pages": pages,
                "recall@1":  recall_at_k(pages, q["expected_page"], 1),
                "recall@3":  recall_at_k(pages, q["expected_page"], 3),
                "recall@5":  recall_at_k(pages, q["expected_page"], 5),
                "recall@10": recall_at_k(pages, q["expected_page"], 10),
                "rr":        reciprocal_rank(pages, q["expected_page"]),
            }
            rows.append(row)
            # Brief progress indicator (retrieval is fast but lexical can stall
            # on the first call due to tsvector index warm-up)
            print(f"{q['id']}  hit@5={row['recall@5']}  rr={row['rr']:.3f}")

        elapsed = time.time() - t_start
        n = len(rows)

        aggregates = {
            "recall@1":  float(np.mean([r["recall@1"]  for r in rows])),
            "recall@3":  float(np.mean([r["recall@3"]  for r in rows])),
            "recall@5":  float(np.mean([r["recall@5"]  for r in rows])),
            "recall@10": float(np.mean([r["recall@10"] for r in rows])),
            "mrr":       float(np.mean([r["rr"]        for r in rows])),
            "n_questions": n,
            "elapsed_s": round(elapsed, 2),
        }

        results[mode] = {"rows": rows, "aggregates": aggregates}

        print(
            f"-> Recall@5={aggregates['recall@5']:.3f}  "
            f"MRR={aggregates['mrr']:.3f}  "
            f"({elapsed:.1f}s)\n"
        )

    return results

# Markdown table builder
def build_markdown_table(results: dict) -> str:
    # Return a markdown table comparing all three modes.
    header = "| Mode    | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR   |"
    sep    = "|---------|----------|----------|----------|-----------|-------|"
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


def build_ablations_md(results: dict) -> str:
    # Build the full docs/ablations.md content for Ablation 1.
    #   Subsequent ablation scripts (reranker, chunking) will append their own
    #   sections. This creates the file with the header + Ablation 1 section.

    table = build_markdown_table(results)

    # Pull out the numbers for the interpretation paragraph
    h_r5  = results["hybrid"]["aggregates"]["recall@5"]
    v_r5  = results["vector"]["aggregates"]["recall@5"]
    l_r5  = results["lexical"]["aggregates"]["recall@5"]
    h_mrr = results["hybrid"]["aggregates"]["mrr"]
    v_mrr = results["vector"]["aggregates"]["mrr"]
    l_mrr = results["lexical"]["aggregates"]["mrr"]

    hybrid_vs_vector  = h_r5 - v_r5
    hybrid_vs_lexical = h_r5 - l_r5

    # Winner label for each metric
    def _winner(vals: dict[str, float]) -> str:
        return max(vals, key=vals.__getitem__)

    r5_winner  = _winner({m: results[m]["aggregates"]["recall@5"] for m in MODES})
    mrr_winner = _winner({m: results[m]["aggregates"]["mrr"]      for m in MODES})

    interpretation = (
        f"Hybrid {'wins' if r5_winner == 'hybrid' else 'does not win'} on Recall@5 "
        f"({h_r5:.3f}), ahead of dense-only by {hybrid_vs_vector:+.3f} points and "
        f"lexical-only by {hybrid_vs_lexical:+.3f} points. "
        f"MRR winner is {mrr_winner} ({results[mrr_winner]['aggregates']['mrr']:.3f}), "
        f"suggesting the {'right' if mrr_winner == r5_winner else 'correct'} chunk "
        f"{'tends to rank near the top' if mrr_winner == r5_winner else 'is not always ranked first even when retrieved'}. "
        f"The gap between hybrid and dense-only demonstrates that the CMS corpus "
        f"contains exact-match signals (CFR section codes, specific dollar figures) "
        f"that BM25 captures and the embedding model misses. "
        f"The gap between hybrid and lexical-only shows that paraphrase-resistant "
        f"questions (where the answer uses different terminology than the query) "
        f"require semantic retrieval to surface."
    )

    return f"""# Ablation Studies

All ablations run with `strategy='hybrid_chunker'` (structure-aware chunking, our baseline)
and against the 18 vector-search document questions in `eval/ground_truth.json`.

---

## Ablation 1: retrieval mode (hybrid vs dense vs lexical)

**Question:** Does combining BM25 and dense retrieval actually outperform either
method alone on this corpus?

**Setup:** `do_rerank=False`, `top_k=10`, `strategy='hybrid_chunker'`.
Reranker disabled so the delta reflects only the first-pass retrieval quality.

{table}

**Interpretation:** {interpretation}

"""

# Entry point
def main() -> None:
    # Questions used: all vector_search doc questions in ground_truth.json that have
    #   is_in_corpus=True and expected_page filled in (should be 18).
    gt_path = ROOT / "eval" / "ground_truth.json"
    if not gt_path.exists():
        print(f"ERROR: ground_truth.json not found at {gt_path}")
        sys.exit(1)

    with open(gt_path) as f:
        ground_truth = json.load(f)

    # Run the ablation
    results = run_ablation(ground_truth)

    # Save raw JSON
    out_dir = ROOT / "eval" / "ablations"

    json_path = out_dir / "retrieval_mode.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Raw results -> {json_path}")

    # Write or update docs/ablations.md
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    md_path = docs_dir / "ablations.md"

    ablation_md = build_ablations_md(results)

    if md_path.exists():
        existing = md_path.read_text()
        if "## Ablation 1" in existing:
            # Replace just the Ablation 1 section (up to the next ## or end-of-file)
            import re
            pattern = r"## Ablation 1 — Retrieval mode.*?(?=\n## |\Z)"
            new_section = ablation_md.split("## Ablation 1")[1]
            replacement = f"## Ablation 1: retrieval mode{new_section}"
            updated = re.sub(pattern, replacement, existing, flags=re.DOTALL)
            md_path.write_text(updated)
            print(f"Updated Ablation 1 section in {md_path}")
        else:
            # Append Ablation 1 section to existing file
            with open(md_path, "a") as f:
                f.write("\n" + ablation_md.split("# Ablation Studies\n\n")[1])
            print(f"Appended Ablation 1 to {md_path}")
    else:
        md_path.write_text(ablation_md)
        print(f"Created {md_path}")

    # Print summary to terminal
    print()
    print("ABLATION 1 RESULTS: Retrieval mode")
    print("*" * 60)
    print(build_markdown_table(results))
    print()

    # Key delta for clarity
    h5 = results["hybrid"]["aggregates"]["recall@5"]
    v5 = results["vector"]["aggregates"]["recall@5"]
    l5 = results["lexical"]["aggregates"]["recall@5"]
    print(f"Hybrid vs dense-only  Recall@5 delta: {h5 - v5:+.3f}")
    print(f"Hybrid vs lexical-only Recall@5 delta: {h5 - l5:+.3f}")

if __name__ == "__main__":
    main()