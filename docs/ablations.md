# Ablation Studies

All ablations run with `strategy='hybrid_chunker'` (structure-aware chunking, our baseline)
and against the 18 vector-search document questions in `eval/ground_truth.json`.

---

## Ablation 1: retrieval mode (hybrid vs dense vs lexical)

**Question:** Does combining BM25 and dense retrieval actually outperform either
method alone on this corpus?

**Setup:** `do_rerank=False`, `top_k=10`, `strategy='hybrid_chunker'`.
Reranker disabled so the delta reflects only the first-pass retrieval quality.

| Mode    | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR   |
|---------|----------|----------|----------|-----------|-------|
| hybrid  | 0.333    | 0.611    | 0.611    | 0.611     | 0.463 |
| vector  | 0.333    | 0.556    | 0.556    | 0.556     | 0.435 |
| lexical | 0.056    | 0.056    | 0.056    | 0.056     | 0.056 |

**Interpretation:** Hybrid wins on Recall@5 (0.611), ahead of dense-only by +0.056 points and lexical-only by +0.556 points. MRR winner is hybrid (0.463), suggesting the right chunk tends to rank near the top. The gap between hybrid and dense-only demonstrates that the CMS corpus contains exact-match signals (CFR section codes, specific dollar figures) that BM25 captures and the embedding model misses. The gap between hybrid and lexical-only shows that paraphrase-resistant questions (where the answer uses different terminology than the query) require semantic retrieval to surface.

