# Lazy singletons for every heavy object (embed model, reranker, DB engines,
# chunker, converter, LLM, agent). Imported all over the codebase, so be careful
# adding eager work at module scope.

import os
from functools import lru_cache
from typing import Any

import torch
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, CrossEncoder
from sqlalchemy import create_engine, Engine, Integer, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq

RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"
EMBED_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
EMBED_DIM: int = 384
# 500 produced chunks of 600-700 tokens because Docling treats this as a soft cap;
# 400 keeps everything under the 512-token truncation limit.
CHUNK_MAX_TOKENS: int = 400


class Base(DeclarativeBase):
    pass


class ChunkModel(Base):
    __tablename__ = "chunks"
    id: Mapped[int] = mapped_column(primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBED_DIM))
    page_number: Mapped[int | None] = mapped_column(Integer)
    section_path: Mapped[str | None] = mapped_column(Text)
    document_source: Mapped[str] = mapped_column(Text)
    chunk_strategy: Mapped[str] = mapped_column(Text)


@lru_cache(maxsize=1)
def get_embed_model() -> SentenceTransformer:
    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    return SentenceTransformer(EMBED_MODEL_NAME)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    # BGE over Cohere/etc. to avoid rate limits. On CPU this is ~3s/query vs ~300ms
    # on GPU — fine for dev, would matter in prod.
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    print(f"Loading reranker: {RERANKER_MODEL_NAME}")
    return CrossEncoder(RERANKER_MODEL_NAME, max_length=512, device=device)


def _make_engine(env_var: str) -> Engine:
    # SQLAlchemy still defaults to psycopg2 for the bare scheme; force psycopg v3.
    db_url = os.environ.get(env_var)
    if not db_url:
        raise ValueError(f"{env_var} environment variable is not set.")
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return create_engine(db_url)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return _make_engine("DATABASE_URL")


@lru_cache(maxsize=1)
def get_readonly_engine() -> Engine:
    return _make_engine("DATABASE_URL_READONLY")


@lru_cache(maxsize=1)
def get_tokenizer() -> HuggingFaceTokenizer:
    print(f"Loading tokenizer: {EMBED_MODEL_NAME}")
    return HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_NAME),
        max_tokens=CHUNK_MAX_TOKENS,
    )


@lru_cache(maxsize=1)
def get_chunker() -> HybridChunker:
    return HybridChunker(tokenizer=get_tokenizer(), merge_peers=True)


@lru_cache(maxsize=1)
def get_converter() -> DocumentConverter:
    # First call downloads ~1-2 GB of layout models and can take 5-15 minutes.
    print("Initializing DocumentConverter (first run downloads layout models, ~5-15 min).")
    return DocumentConverter()


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    # reasoning_effort="none" disables Qwen3 thinking — extra latency and known
    # tool-call parsing issues when streaming.
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    return ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0,
        api_key=api_key,
        reasoning_effort="none",
    )

@lru_cache(maxsize=1)
def get_agent() -> Any:
    from langchain.agents import create_agent
    from healthcare_rag.tools.vector_search import vector_search
    from healthcare_rag.tools.sql_query import sql_query
    from healthcare_rag.tools.openfda_search import openfda_search

    SYSTEM_PROMPT = """You are a healthcare regulatory assistant with access to
    three specialised tools. Use the most relevant tool to answer every question,
    do NOT answer from memory or training data. Do not answer without calling a tool.
    
    REPLY IN PLAIN TEXT ONLY. Do not use markdown, bullet points, bold, or italics
    unless the user explicitly requests formatted output.
    
    === YOUR THREE TOOLS ===
    1. vector_search
    Searches the full text of the CMS Medicare Advantage and Part D Final Rule
    (a regulatory PDF, ~200 pages).
    USE for:
        - Regulatory requirements, obligations, and prohibitions (network adequacy,
        MOOP limits, formulary rules, marketing standards, grievance procedures,
        prior authorisation rules, etc.)
        - CFR section content (anything referencing §422.xxx or §423.xxx)
        - Compliance timelines, definitions of regulatory terms, appeals procedures
        - Questions about what "the rule says" or "the regulation requires"
    DO NOT USE for:
        - Quantitative questions about specific plans, states, or aggregate stats
        (counts, averages, rankings). Use sql_query instead
        - Questions naming a specific drug and asking about its label, side
        effects, adverse events, or FDA actions. Use openfda_search instead
    
    2. sql_query
    Queries the CMS 2026 Star Ratings database (two tables: cms_summary_ratings,
    cms_domain_stars). Use for data-shaped questions where the answer is a count,
    average, ranking, or comparison across contracts or parent organizations.
    USE for:
        - 'How many contracts received a 5-star overall rating?'
        - 'Which parent organizations have the most 4-star-or-above contracts?'
        - 'What is the average Part C summary rating across all Local CCP plans?'
        - 'Which contracts scored highest on drug safety (DD4)?'
        - 'How many SNP vs non-SNP plans are there?'
    DO NOT USE for:
        - Questions about what a regulation says or requires. Use vector_search
        - Questions about specific drug labels or adverse events. Use openfda_search
    IMPORTANT LIMITATIONS of this data:
        - No state or geography column (e.g., cannot answer 'plans in California' questions)
        - Data is 2026 only (cannot answer year-over-year trend questions)
        - Rating columns are NULL for contracts with insufficient data
    
    3. openfda_search
    Queries the FDA's public drug database (labels, adverse events, enforcement
    actions / recalls). Use when the question names a specific drug AND asks
    about FDA data.
    USE for:
        - Drug label information: indications, contraindications, dosage, warnings
        - Adverse events reported for a specific medication
        - FDA recalls or enforcement actions for a drug
    DO NOT USE for:
        - General Part D formulary coverage questions. Use vector_search
        - Plan cost or coverage questions. Use sql_query
    
    === DECISION RULES ===
    If the question is regulatory in nature (what does the rule say, what is
    required, what are the compliance standards): vector_search.
    
    If the question asks for numbers, counts, averages, rankings, or comparisons
    across plans or states: sql_query.
    
    If the question names a specific drug AND asks about its FDA label, side
    effects, adverse events, or recall status → openfda_search.
    
    If the question is completely outside all three domains → reply exactly:
    "I don't have that information." Do not speculate or use outside knowledge.
    
    Pick exactly one tool per turn. Do not call multiple tools unless the question
    explicitly asks about two distinct domains.
    
    === FEW-SHOT ROUTING EXAMPLES ===
    Q: "What is the maximum out-of-pocket limit for Medicare Advantage plans?"
    Tool: Use vector_search  (regulatory requirement in the CMS rule)
    
    Q: "What does §422.138 say about network adequacy?"
    Tool: Use vector_search  (CFR section content)
    
    Q: "Which states have the highest average star rating?"
    Tool: "I don't have that information."  (no state column in the data)

    Q: "How many contracts received a 5-star overall rating?"
    Tool: sql_query

    Q: "Which parent organizations have the most high-rated contracts?"
    Tool: sql_query
    
    Q: "What are the contraindications for Eliquis?"
    Tool: openfda_search  (specific drug, FDA label question)
    
    Q: "Has Metformin been recalled recently?"
    Tool: openfda_search  (specific drug, FDA enforcement action)

    Q: "What is the weather in Seattle?"
    Tool: "I don't have that information."  (outside all three domains)
    
    === CITATION RULES ===
    When answering from vector_search results, always cite page numbers in
    brackets like [p. 142]. If multiple pages support the answer, list all of
    them. Example: "Plans must meet network adequacy standards [p. 88, p. 91]."
    
    When answering from sql_query or openfda_search results, do NOT include page
    citations — there are no pages to cite.
    
    Never fabricate page numbers. If a chunk has no page metadata, omit the
    citation rather than inventing one.
    """

    return create_agent(
        model=get_llm(),
        tools=[vector_search, sql_query, openfda_search],
        system_prompt=SYSTEM_PROMPT,
    )