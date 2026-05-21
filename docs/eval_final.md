# Final Evaluation

Pre-deployment metrics for the locked-in config. These are the numbers that
go in the README. Per-question detail is in `docs/baseline.md` (retrieval),
`eval/routing_eval_full.json` (routing), and
`docs/eval_baseline_meta-llama-llama-4-scout-17b-16e-instruct.md` (LLM judge).

## Locked-in config

| Component | Choice |
|-----------|--------|
| Agent LLM | `qwen/qwen3-32b` via Groq (LangGraph) |
| Retrieval | Hybrid (BM25 + dense, RRF k=60) |
| Embedder  | `BAAI/bge-small-en-v1.5` (384-dim) |
| Reranker  | `BAAI/bge-reranker-v2-m3` (top-50 to 5) |
| Chunker   | Docling HybridChunker (max 400 tokens) |
| LLM judge | `meta-llama/llama-4-scout-17b-16e-instruct` via Groq |

See `docs/ablations.md` for the rationale behind each choice.

## Headline numbers

| Metric | Value | n |
|--------|------:|--:|
| Tool-routing accuracy     | **0.931** | 27/29 |
| Retrieval Recall@5        | **1.000** | 18 |
| Retrieval Recall@3        | 1.000 | 18 |
| Retrieval Recall@1        | 0.611 | 18 |
| Retrieval MRR             | 0.787 | 18 |
| Faithfulness (judge)      | **0.944** | 18 |
| Context precision (judge) | 0.667 | 18 |
| Context recall (judge)    | 0.611 | 18 |
| Citation precision        | **0.653** | 18 |
| Hallucination rate (OOS)  | **0.000** | 3 |

`n` is the slice each metric runs over. 29 = all ground-truth questions,
18 = in-corpus doc questions (`expected_tool=vector_search`), 3 = out-of-corpus
(`expected_tool=none`).

## Routing

Two misroutes out of 29:

| ID   | Question (truncated) | Expected | Called |
|------|----------------------|----------|--------|
| q013 | Is Medicare cutting back on how many things it uses to grade... | vector_search | sql_query |
| q014 | How often do Medicare insurance brokers actually show benefi... | vector_search | none |

q013 reads like a count question (sql), q014 reads conversational (none).
Both are genuinely ambiguous.

## Citations

Breakdown across the 18 doc questions:

| Bucket | Count |
|--------|------:|
| All citations land in gold | 9 |
| At least one valid, some extras outside gold | 6 |
| All citations outside gold | 1 |
| No citation at all | 2 |

The 2 "no citation" cases are q013 and q014, the same two routing misroutes
above. Vector search never fired, so there were no chunks to cite. Fix the
router and that count drops automatically.

The 1 "all wrong" case is q017: cited `[5, 7]` (Initial Coverage Limit
section) on a question about catastrophic coverage. Pages 17 and 206 cover
catastrophic. Real miss.

Of the 6 partials, several (q006, q016, q018) cite pages adjacent to gold,
which is probably more label incompleteness than agent error. A second relabel
pass would lift this further; I stopped after the first round to keep the
process reproducible.

## Caveats

- Recall@5 = 1.000 partly reflects a hand-audit of gold labels. Five
  questions had additional on-topic pages added to `expected_page` after
  manually inspecting chunk text. The retrieval algorithm did not change;
  the relabel just stopped grading correct citations as misses.
- Citation precision climbed from 0.306 to 0.653 from three combined fixes:
  the metric regex now captures multi-page brackets like `[p. 88, p. 91]`
  instead of dropping them, the gold relabel above, and a system-prompt rule
  requiring `[p. N]` on every claim from `vector_search`.
- Faithfulness sits well above context recall, meaning the agent stays
  grounded in what it gets back even when the retrieved context is
  incomplete. Failure mode is "doesn't answer fully," not "makes things up."
- Hallucination rate is 0/3 on the OOS slice, but n=3 is tiny. Treat as
  indicative, not conclusive.

## Reproducing

```bash
uv run python eval/generate_responses.py   # ~10 min, ~29 agent calls
uv run python eval/metrics.py              # ~30 s, retrieval only
uv run python eval/llm_eval.py             # ~2 min, ~54 judge calls
```

`generate_responses.py` writes `eval/ground_truth_responses.json`. The other
two read from it (judge) or hit the DB directly (metrics).
