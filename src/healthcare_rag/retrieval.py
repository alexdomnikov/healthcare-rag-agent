# Three retrieval strategies, all returning list[RetrievedChunk]:
#   retrieve_dense (pgvector cosine similarity)
#   retrieve_lexical (Postgres tsvector / ts_rank_cd : BM25 variant)
#   retrieve_hybrid  (Reciprocal rank fusion of both)

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from healthcare_rag.core import get_embed_model, get_engine

# Important constants
DEFAULT_STRATEGY = "structure" # swap to "fixed" for ablation studies
RRF_K = 60 # canonical value from Cormack et al. (2009), tends to perform best
FIRST_PASS_K = 100 # candidates fed into RRF before final top-k cut

@dataclass
class RetrievedChunk:
    id: int
    text: str
    page_number: int | None
    section_path: str | None
    score: float  # rrf_score, cosine similarity, or ts_rank depending on method

def embed_query(query:str):
    # Embed a user query with BGE's query-instruction prefix, normalized.

    return get_embed_model().encode(
        query,
        prompt_name="query", # BGE requires this prefix at query time
        normalize_embeddings=True,
    ).tolist()

def retrieve_dense(
    query:str,
    top_k=50,
    strategy=DEFAULT_STRATEGY,
):
    # Dense retrieval: embed the query, find chunks with similar embeddings. 
    #   Captures semantic similarity rather than exact matches.
    
    query_vec = embed_query(query)

    # The <=> operator is pgvector's cosine *distance* (smaller = more similar),
    #   so we ORDER BY ascending distance and convert to similarity score 
    #   with 1 - distance.
    sql = text("""
        SELECT
            id,
            text,
            page_number,
            section_path,
            1 - (embedding <=> :qv::vector) AS score
        FROM chunks
        WHERE chunk_strategy = :strategy
        ORDER BY embedding <=> :qv::vector
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
    query:str,
    top_k=50,
    strategy=DEFAULT_STRATEGY,
):
    # Score documents by how many query terms they contain, weighted by term 
    #   rarity (accounts for term frequency & inverse document frquency) and
    #   document length. Fast, interpretable, use for exact-match.

    # NOTE: plainto_tsquery is used (not to_tsquery) because it safely handles
    #   arbitrary natural-language input without requiring the user to write
    #   query operators like & or |.
    # NOTE: Chunks with zero lexical overlap are excluded by the @@ filter. They'd
    #   score 0 anyway, so excluding them speeds the query on large tables.
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
    query:str,
    top_k=50,
    strategy=DEFAULT_STRATEGY,
    k=RRF_K,
):
    # Fuse dense and lexical ranked lists with Reciprocal Rank Fusion (RRF).

    query_vec = embed_query(query)
    
    if not query_vec:
        return []

    # RRF score for a document d = sigma  1 / (k + rank_i(d))
    #    for all i belonging to {dense, lexical}, where k=60 is the canonical
    #    constant recommended from Cormack et all. (2009).

    # Everything happens in a single SQL query using CTEs:
    #   dense: top-FIRST_PASS_K results ordered by cosine distance
    #   lexical: top-FIRST_PASS_K results ordered by ts_rank_cd
    #   fused: UNION ALL -> GROUP BY id -> SUM(1/(k+rank))
    # NOTE: a single query avoids two round-trips to Neon and lets Postgres
    #    execute both sub-plans in one connection context.
    sql = text("""
        WITH dense AS (
            SELECT
                id,
                ROW_NUMBER() OVER (ORDER BY embedding <=> :qv::vector) AS rank
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
                "qv":         str(query_vec),
                "q":          query,
                "strategy":   strategy,
                "first_pass": FIRST_PASS_K,
                "k":          k,
                "top_k":      top_k,
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