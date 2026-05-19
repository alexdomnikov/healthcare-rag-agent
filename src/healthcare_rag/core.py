# Shared resources for the project. Every heavy object (embed model, DB engine,
#   chunker, document converter) lives here as a lazy singleton.

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
EMBED_DIM: int = 384 # 384 matches BGE-small's output
CHUNK_MAX_TOKENS: int = 400 # 500 had many chunks w/ 600-700 tokens, truncation starts at 512

#*****************************************************************************
# INGESTION, RETRIEVAL
#*****************************************************************************
class Base(DeclarativeBase):
    # ORM model.
    pass

class ChunkModel(Base):
    # Tells SQLAlchemy what the db table layout is.
    
    __tablename__ = "chunks"
    # Mapped[type] gives strict Python typing, mapped_column configures the DB
    id: Mapped[int] = mapped_column(primary_key=True)
    text: Mapped[str] = mapped_column(Text)

    # Pull directly from EMBED_DIM to ensure everything matches
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBED_DIM))

    page_number: Mapped[int | None] = mapped_column(Integer)
    section_path: Mapped[str | None] = mapped_column(Text)
    document_source: Mapped[str] = mapped_column(Text)
    chunk_strategy: Mapped[str] = mapped_column(Text)

@lru_cache(maxsize=1)
def get_embed_model() -> SentenceTransformer:
    # Return the shared sentence-transformer embedding model.

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    # NOTE: normalize_embeddings isn't applied here; callers pass at encode() time.
    return SentenceTransformer(EMBED_MODEL_NAME)

@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    # Return the shared BGE cross-encoder reranker. 
    # NOTE: I chose BGE over a managed API (Cohere, etc.) mostly to avoid rate limits.

    # Check if GPU is available, would be ~4-10x faster. Negligible in this context
    #   since we're talking maybe 3 seconds vs. 300 milliseconds, but this would be
    #   critical in a production context (with the assumption that an API like
    #   Cohere is still not chosen as an alternative).
    if torch.cuda.is_available():
        device = 'cuda' 
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    
    # A cross-encoder takes (query, passage) as a single concatenated input and
    #   outputs one relevance score. More accurate than cosine similarity between
    #   bi-encoder embeddings because the model attends to both texts simultaneously
    #   at the cost of not being precomputable (i.e., computation heavy).
    # max_length=512 matches BGE-small's context window; longer inputs truncated.
    print(f"Loading reranker: {RERANKER_MODEL_NAME}")
    return CrossEncoder(RERANKER_MODEL_NAME, max_length=512, device=device)

@lru_cache(maxsize=1)
def get_engine() -> Engine:
    # Return the shared SQLAlchemy engine connected to Neon Postgres.
 
    # NOTE: Rewrites scheme from 'postgresql://' to 'postgresql+psycopg://' 
    #   so SQLAlchemy uses the psycopg v3 driver instead of psycopg2.
    db_url = os.environ.get("DATABASE_URL")
    
    if not db_url:
        raise ValueError("DATABASE_URL environment variable is not set.")
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    
    return create_engine(db_url)

@lru_cache(maxsize=1)
def get_readonly_engine() -> Engine:
    # Return the shared SQLAlchemy engine connected to Neon Postgres w/ readonly role.
 
    db_url = os.environ.get("DATABASE_URL_READONLY")
    
    if not db_url:
        raise ValueError("DATABASE_URL_READONLY environment variable is not set.")
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    
    return create_engine(db_url)

@lru_cache(maxsize=1)
def get_tokenizer() -> HuggingFaceTokenizer:
    # Return the shared HuggingFace tokenizer wrapper used by the chunker.

    print(f"Loading tokenizer: {EMBED_MODEL_NAME}")
    return HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_NAME),
        max_tokens=CHUNK_MAX_TOKENS,
    )

@lru_cache(maxsize=1)
def get_chunker() -> HybridChunker:
    # Return the shared HybridChunker instance.
    
    return HybridChunker(tokenizer=get_tokenizer(), merge_peers=True)

@lru_cache(maxsize=1)
def get_converter() -> DocumentConverter:
    # Return the shared Docling DocumentConverter.
 
    # NOTE: First call downloads layout models (~1-2 GB) and can take 5-15 minutes.
    #   Subsequent calls return immediately from cache.
    print("Initializing DocumentConverter (Layout models downloaded on first run, ~5-15 mins).")
    return DocumentConverter()

#*****************************************************************************
# AGENT-RELATED
#*****************************************************************************
@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    # Return the shared Groq LLM.
    
    # reasoning_effort="none" disables Qwen3's thinking mode. This mode adds 
    #   latency and has known tool-call parsing issues when streaming.
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
    from healthcare_rag.tools.vector_search import vect_search_tool
    from healthcare_rag.tools.sql_query import sql_query_tool

    SYSTEM_PROMPT = """You are a healthcare regulatory assistant. Answer 
    questions using the available tools. REPLY IN PLAIN TEXT ONLY; do not 
    use markdown formatting, bold, italics, or bullet points unless the user 
    explicitly asks for formatted output.
    
    When answering from regulatory text retrieved by vector_search, always 
    cite page numbers in brackets like [p. 142]; if multiple pages support 
    your answer, then list all of them. When answering from sql_query or 
    openfda_search results, do NOT include page citations as there are no 
    pages to cite.
    
    DO NOT ANSWER BEFORE SELECTING AND CALLING THE MOST RELEVANT TOOL ACCORDING 
    TO TOOL DESCRIPTIONS (i.e., ALWAYS TRY CALLING A TOOL BEFORE ANSWERING). 
    If the selected tool returns no relevant context, then say exactly "I don't 
    have that information." Do not speculate, use outside knowledge, or invent 
    citations. Always use specified tools to answer questions. Do not answer 
    from memory.
    """

    return create_agent(
        model=get_llm(),
        tools=[vect_search_tool, sql_query_tool],
        system_prompt=SYSTEM_PROMPT,
    )