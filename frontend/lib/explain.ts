export const EXPLAIN_TRIGGER = "EXPLAIN";
export const CLEAR_TRIGGER = "CLEAR";

export const EXPLAIN_CONTENT = `Thanks for trying the demo. Below is a quick tour of what's under the hood, the numbers I measured, and what I took away from building it.

WHAT IT IS

A RAG agent that routes questions across three healthcare data sources:
  - The CMS Medicare Advantage and Part D Final Rule, Contract Year 2027 (~200-page regulatory PDF)
  - The CMS 2026 Star Ratings Data Tables (Postgres, two tables)
  - The openFDA public API (drug labels, adverse events, recalls)

The agent picks one of three tools per question (vector_search, sql_query, openfda_search) and answers from retrieved evidence. Answers from the regulatory PDF include page-level citations.

ARCHITECTURE

Ingestion:
  - Docling for structure-aware PDF parsing (preserves section hierarchy and page numbers)
  - HybridChunker with a HuggingFace tokenizer, 400-token cap
  - BAAI/bge-small-en-v1.5 embeddings, stored in Postgres + pgvector

Retrieval:
  - Hybrid search: dense (pgvector) plus lexical (Postgres FTS), fused with Reciprocal Rank Fusion
  - Cross-encoder reranking with BAAI/bge-reranker-v2-m3

Agent:
  - LangChain 1.0 + LangGraph, three tools, single-shot routing per turn
  - Qwen3-32B on Groq for fast tool calling
  - LangSmith tracing on every call

Serving:
  - FastAPI with SSE streaming
  - Per-IP rate limiting, a request timeout, and graceful handling of Groq 429s (both TPM and TPD)
  - Next.js frontend with token-level streaming

THE NUMBERS

Retrieval ablation (N=18, Recall@5 on the CMS rule corpus, no reranker, to isolate first-pass quality):
  - Hybrid (RRF):     61.1%
  - Dense only:       55.6%
  - Lexical only:      5.6%

MRR: 0.46 hybrid vs 0.44 dense. Hybrid wins on both top-of-list quality and tail recall.

Shipped retrieval (hybrid + cross-encoder rerank) end-to-end: Recall@5 = 0.83, MRR = 0.66. The reranker adds +22 points of Recall@5 over the raw hybrid stage.

Routing accuracy: 93.1% (27/29) on a labeled tool-routing set.

LLM-as-judge evaluation (meta-llama/llama-4-scout-17b-16e-instruct via Groq, N=18 in-corpus doc questions):
  - Faithfulness:        0.94
  - Context precision:   0.67
  - Context recall:      0.61
  - Citation precision:  0.65
  - Hallucination rate:  0/3 OOS (n=3, indicative only)

Judge metrics jitter ±0.05 between runs; treat as approximate.

WHAT I LEARNED

1. Before I had numbers I would have shipped dense-only retrieval and assumed it was fine. Running the ablation showed hybrid retrieval gains 5.5 points on Recall@5. Modest on its own, but hybrid also feeds the reranker better candidates, and the reranker is where the real gain is (+22.2 points). The better first pass impacts this positively.

2. Pure lexical retrieval performs badly on regulatory text (5.6% Recall@5). Users phrase questions in plain English, but the regulation uses precise legal language, so the keywords rarely overlap. Lexical still helps when fused with dense, especially when a question contains a specific CFR section number or a unique phrase.

3. Chunking strategy had a non-obvious trade-off. Fixed-size 512-token chunks scored +5.6 points higher on Recall@5 than Docling's structure-aware HybridChunker (400-token cap, merge_peers enabled). But structure-aware chunks carry section_path metadata that produces noticeably cleaner citations downstream. I kept structure-aware in production. Citation quality is what users see, and the recall trade is small enough that the reranker absorbs it.

4. Tool routing is most of the work and most of the bugs. The model picks the right tool 93% of the time. The 7% that misroute are usually genuinely ambiguous questions where I'd second-guess myself too. Adding a few-shot routing block to the system prompt was the single biggest accuracy bump.

5. A lot of the polish at the end had nothing to do with retrieval: structured request logs, LangSmith traces, per-IP rate limiting, request timeouts, graceful upstream rate-limit handling, environment-aware CORS, no exception details in error responses, and a click-through disclaimer.

STACK

Python, FastAPI, LangChain 1.0, LangGraph, Postgres (Neon) + pgvector, Docling, sentence-transformers, Groq (Qwen3-32B), LangSmith, Next.js, TypeScript, Tailwind.

MORE

Repo: https://github.com/alexdomnikov/healthcare-rag-agent

To contact me, email me@alexdomnikov.com.

Type any question to try the agent (e.g., 'What are the contraindications for Eliquis?').`;