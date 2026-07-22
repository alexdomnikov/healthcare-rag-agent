export const EXPLAIN_TRIGGER = "EXPLAIN";
export const CLEAR_TRIGGER = "CLEAR";

export const EXPLAIN_CONTENT = `Thanks for trying the demo. Here's a short tour of what's under the hood.

WHAT IT IS

A RAG agent that routes each question to one of three healthcare sources:
  - The CMS Medicare Advantage and Part D Final Rule, Contract Year 2027 (~200-page regulatory PDF)
  - The CMS 2026 Star Ratings data tables (Postgres)
  - The openFDA public API (drug labels, adverse events, recalls)

It picks exactly one tool per question (vector_search, sql_query, openfda_search) and answers from retrieved evidence. Answers from the regulatory PDF carry page-level citations.

HOW IT WORKS

Ingestion: Docling structure-aware PDF parsing, 400-token HybridChunker, bge-small-en-v1.5 embeddings in Postgres + pgvector.

Retrieval: hybrid dense + lexical search fused with Reciprocal Rank Fusion, then cross-encoder reranking with bge-reranker-v2-m3 (top-50 down to top-5).

Agent: LangChain 1.0 + LangGraph, three tools, gpt-oss-120b on Groq, LangSmith tracing on every call.

Serving: FastAPI with SSE streaming, per-IP rate limiting, request timeouts, and graceful handling of Groq 429s. Next.js frontend with token-level streaming.

THE NUMBERS

Measured over 29 hand-written questions (18 in-corpus document questions, 3 out-of-corpus):

  Tool-routing accuracy      1.000 (29/29)
  Retrieval Recall@5         1.000
  Retrieval Recall@1         0.611
  Retrieval MRR              0.787
  Faithfulness               0.750
  Context precision          0.861
  Context recall             0.639
  Citation precision         0.668
  Hallucination rate         0.000 (n=3, indicative only)

Faithfulness and the context metrics come from a custom async LLM judge built on qwen/qwen3.6-27b in reasoning mode, deliberately a different model family from the agent to avoid same-family self-preference bias.

WHAT I LEARNED

1. The reranker is where retrieval is won. Hybrid retrieval alone lands Recall@5 at 0.778; the cross-encoder takes it to 1.000. Hybrid's real job is feeding the reranker good candidates, not being right on its own.

2. Pure lexical retrieval is close to useless on regulatory text (Recall@5 = 0.056). People ask in plain English while the regulation is written in precise legal language, so the keywords rarely overlap. Lexical still earns its place in the fusion when a question names a CFR section or a dollar figure.

3. Fixed-size chunking beat structure-aware chunking on raw recall, but I shipped structure-aware anyway: it carries section metadata that produces cleaner citations, and the reranker absorbs the difference. The numerical winner isn't automatically the right thing to ship.

4. Faithfulness (0.750) sitting above context recall (0.639) is the most useful number here. The agent stays grounded in what it retrieves even when retrieval comes back incomplete, so its failure mode is "doesn't answer fully," not "makes things up."

5. Most of the eval work was fixing the eval, not the agent. A too-lenient judge had inflated faithfulness to 0.94, and a citation extractor that only understood ASCII brackets was silently dropping a third of the model's citations. Measure the measurement first.

STACK

Python, FastAPI, LangChain 1.0, LangGraph, Postgres (Neon) + pgvector, Docling, sentence-transformers, Groq (gpt-oss-120b), LangSmith, Next.js, TypeScript, Tailwind.

MORE

Repo: https://github.com/alexdomnikov/healthcare-rag-agent
Contact: me@alexdomnikov.com

Type any question to try the agent (e.g., 'What are the contraindications for Eliquis?').`;
