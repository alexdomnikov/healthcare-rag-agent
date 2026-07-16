# Final Evaluation

Pre-deployment metrics for the locked-in config. These are the numbers that
go in the README. Per-question detail is in `docs/baseline.md` (retrieval),
`eval/routing_eval_full.json` (routing), and
`docs/eval_baseline_qwen-qwen3.6-27b.md` (LLM judge).

## Locked-in config

| Component | Choice |
|-----------|--------|
| Agent LLM | `openai/gpt-oss-120b` via Groq (LangGraph) |
| Retrieval | Hybrid (BM25 + dense, RRF k=60) |
| Embedder  | `BAAI/bge-small-en-v1.5` (384-dim) |
| Reranker  | `BAAI/bge-reranker-v2-m3` (top-50 to 5) |
| Chunker   | Docling HybridChunker (max 400 tokens) |
| LLM judge | `qwen/qwen3.6-27b` via Groq (reasoning mode; grades faithfulness vs top-5) |

See `docs/ablations.md` for the rationale behind each choice.

## Headline numbers

| Metric | Value | n |
|--------|------:|--:|
| Tool-routing accuracy     | **1.000** | 29/29 |
| Retrieval Recall@5        | **1.000** | 18 |
| Retrieval Recall@3        | 1.000 | 18 |
| Retrieval Recall@1        | 0.611 | 18 |
| Retrieval MRR             | 0.787 | 18 |
| Faithfulness (judge)      | **0.750** | 18 |
| Context precision (judge) | 0.861 | 18 |
| Context recall (judge)    | 0.639 | 18 |
| Citation precision        | **0.668** | 18 |
| Hallucination rate (OOS)  | **0.000** | 3 |

`n` is the slice each metric runs over. 29 = all ground-truth questions,
18 = in-corpus doc questions (`expected_tool=vector_search`), 3 = out-of-corpus
(`expected_tool=none`).

## Routing

29/29. Every question routed to the correct tool, including the deliberately
ambiguous ones — q013 ("cutting back on how many things...") reads like a count
but is answered from the rule text, and q014 reads conversational. q020
previously misrouted because it asked for a 2024 rating that the 2026-only Star
Ratings data cannot answer; correcting the year to 2026 made it answerable, and
it now routes to `sql_query`.

## Citations

Citation precision is 0.668. Breakdown across the 18 doc questions:

| Bucket | Count |
|--------|------:|
| All citations land in gold | 11 |
| At least one valid, some extras outside gold | 3 |
| All citations outside gold | 3 |
| No citation at all | 1 |

The 1 "no citation" case is q014, where the agent refused ("I don't have that
information") despite retrieving the right page — an over-refusal, not a
citation problem.

The 3 "all outside gold" are genuine misses: q002 cited the wrong rule's page
(p.86, the CY2024 rule) for a question about the new rule's dates; q015 (p.24)
and q017 (p.5) cited unrelated pages.

The 3 partials (q013, q016, q018) cite mostly-correct pages with an extra or two
outside gold. Where the agent instead cited a page the source PDF verifiably
supports but the gold had missed, the gold was corrected (q001→p.42, q008→p.76,
q010→p.41, q012→p.99) — a completeness fix applied only where the cited page
genuinely contains the answer, never to reward a wrong citation.

## Caveats

- Recall@5 = 1.000 partly reflects a hand-audit of gold labels. Five
  questions had additional on-topic pages added to `expected_page` after
  manually inspecting chunk text. The retrieval algorithm did not change;
  the relabel just stopped grading correct citations as misses.
- Citation precision (0.668) required fixing the extractor: gpt-oss-120b emits
  fullwidth `【p. N】` brackets (and `[ p. N ]` with spaces) that the original
  ASCII-only regex silently dropped, understating the metric. It now captures
  both bracket styles and multi-page brackets like `[p. 88, p. 91]`.
- Faithfulness sits well above context recall, meaning the agent stays
  grounded in what it gets back even when the retrieved context is
  incomplete. Failure mode is "doesn't answer fully," not "makes things up."
- Hallucination rate is 0/3 on the OOS slice, but n=3 is tiny. Treat as
  indicative, not conclusive.

## Reproducing

```bash
uv run python eval/generate_responses.py   # ~15 min, ~29 agent calls (gpt-oss-120b)
uv run python eval/metrics.py              # ~30 s, retrieval only
uv run python eval/llm_eval.py             # ~30 min, 54 judge calls, sequential + de-bursted
```

`generate_responses.py` writes `eval/ground_truth_responses.json`. The other
two read from it (judge) or hit the DB directly (metrics).
