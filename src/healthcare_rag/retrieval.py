import os
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session
from langsmith import traceable

from healthcare_rag.core import get_embed_model, get_engine, get_reranker

DEFAULT_STRATEGY = "hybrid_chunker"  # "fixed" for ablation studies
RRF_K = 60
FIRST_PASS_K = 50


@dataclass
class RetrievedChunk:
    id: int
    text: str
    page_number: int | None
    section_path: str | None
    score: float  # rrf, cosine, or ts_rank depending on method


def embed_query(query: str) -> list[float]:
    # BGE requires the "query" prompt prefix at query time; normalized for cosine.
    return get_embed_model().encode(
        query,
        prompt_name="query",
        normalize_embeddings=True,
    ).tolist()


def retrieve_dense(
    query: str,
    top_k: int = 50,
    strategy: str = DEFAULT_STRATEGY,
) -> list[RetrievedChunk]:
    query_vec = embed_query(query)

    # pgvector's <=> is cosine *distance*, so ORDER BY ascending and flip to
    # similarity with 1 - distance.
    sql = text("""
        SELECT
            id,
            text,
            page_number,
            section_path,
            1 - (embedding <=> CAST(:qv AS Vector)) AS score
        FROM chunks
        WHERE chunk_strategy = :strategy
        ORDER BY embedding <=> CAST(:qv AS Vector)
        LIMIT :k
    """)

    with Session(get_engine()) as session:
        rows = session.execute(
            sql,
            {"qv": str(query_vec), "strategy": strategy, "k": top_k},
        ).all()

    return [
        RetrievedChunk(
            id=r.id,
            text=r.text,
            page_number=r.page_number,
            section_path=r.section_path,
            score=float(r.score),
        )
        for r in rows
    ]

def retrieve_lexical(
    query: str,
    top_k: int = 50,
    strategy: str = DEFAULT_STRATEGY,
) -> list[RetrievedChunk]:
    # plainto_tsquery (not to_tsquery) so we don't have to escape user input.
    # The @@ filter drops zero-overlap chunks — they'd score 0 anyway but the
    # filter speeds the query on large tables.
    sql = text("""
        SELECT
            id,
            text,
            page_number,
            section_path,
            ts_rank_cd(tsv, plainto_tsquery('english', :q)) AS score
        FROM chunks
        WHERE chunk_strategy = :strategy
          AND tsv @@ plainto_tsquery('english', :q)
        ORDER BY score DESC
        LIMIT :k
    """)

    with Session(get_engine()) as session:
        rows = session.execute(
            sql,
            {"q": query, "strategy": strategy, "k": top_k},
        ).all()

    return [
        RetrievedChunk(
            id=r.id,
            text=r.text,
            page_number=r.page_number,
            section_path=r.section_path,
            score=float(r.score),
        )
        for r in rows
    ]

def retrieve_hybrid(
    query: str,
    top_k: int = 50,
    strategy: str = DEFAULT_STRATEGY,
    k: int = RRF_K,
) -> list[RetrievedChunk]:
    query_vec = embed_query(query)

    # Both sub-rankings and the RRF fuse happen in one query — one round-trip to
    # Neon, and Postgres runs both sub-plans in the same connection context.
    sql = text("""
        WITH dense AS (
            SELECT
                id,
                ROW_NUMBER() OVER (ORDER BY embedding <=> CAST(:qv AS Vector)) AS rank
            FROM chunks
            WHERE chunk_strategy = :strategy
            LIMIT :first_pass
        ),
        lexical AS (
            SELECT
                id,
                ROW_NUMBER() OVER (
                    ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', :q)) DESC
                ) AS rank
            FROM chunks
            WHERE chunk_strategy  = :strategy
              AND tsv @@ plainto_tsquery('english', :q)
            LIMIT :first_pass
        ),
        fused AS (
            SELECT
                id,
                SUM(1.0 / (:k + rank)) AS rrf_score
            FROM (
                SELECT * FROM dense
                UNION ALL
                SELECT * FROM lexical
            ) combined
            GROUP BY id
        )
        SELECT
            c.id,
            c.text,
            c.page_number,
            c.section_path,
            f.rrf_score
        FROM fused f
        JOIN chunks c ON c.id = f.id
        ORDER BY f.rrf_score DESC
        LIMIT :top_k
    """)

    with Session(get_engine()) as session:
        rows = session.execute(
            sql,
            {
                "qv": str(query_vec),
                "q": query,
                "strategy": strategy,
                "first_pass": FIRST_PASS_K,
                "k": k,
                "top_k": top_k,
            },
        ).all()

    return [
        RetrievedChunk(
            id=r.id,
            text=r.text,
            page_number=r.page_number,
            section_path=r.section_path,
            score=float(r.rrf_score),
        )
        for r in rows
    ]

def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int = 5,
) -> list[RetrievedChunk]:
    # Rerank only the first-pass candidates, not the full corpus — keeps latency
    # ~1-3s instead of scaling with table size.
    if not candidates:
        return []

    model = get_reranker()
    pairs = [(query, chunk.text) for chunk in candidates]
    scores: list[float] = model.predict(pairs, show_progress_bar=False).tolist()
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    return [
        RetrievedChunk(
            id=chunk.id,
            text=chunk.text,
            page_number=chunk.page_number,
            section_path=chunk.section_path,
            score=score,
        )
        for chunk, score in ranked[:top_k]
    ]


@traceable
def retrieve(
    query: str,
    top_k: int = 5,
    do_rerank: bool = True,
    mode: str = "hybrid",
    strategy: str = DEFAULT_STRATEGY,
) -> list[RetrievedChunk]:
    # CI runs on CPU runners where the cross-encoder is the bottleneck. Set
    # DISABLE_RERANKER=1 to short-circuit reranking globally (eval + agent tool
    # path both go through this function). Local/prod still rerank by default.
    if os.getenv("DISABLE_RERANKER") == "1":
        do_rerank = False

    first_pass_k = FIRST_PASS_K if do_rerank else top_k

    if mode == "hybrid":
        candidates = retrieve_hybrid(query, top_k=first_pass_k, strategy=strategy)
    elif mode == "vector":
        candidates = retrieve_dense(query, top_k=first_pass_k, strategy=strategy)
    elif mode == "lexical":
        candidates = retrieve_lexical(query, top_k=first_pass_k, strategy=strategy)
    else:
        raise ValueError(f"Unknown mode {mode!r}. Use 'hybrid', 'vector', or 'lexical'.")

    if not candidates:
        return []

    if do_rerank:
        return rerank(query, candidates, top_k=top_k)

    return candidates[:top_k]