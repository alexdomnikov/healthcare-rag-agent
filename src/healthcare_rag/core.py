# Shared resources for the project. Every heavy object (embed model, DB engine,
#   chunker, document converter) lives here as a lazy singleton.

import os
from functools import lru_cache

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384 # 384 matches BGE-small's output
CHUNK_MAX_TOKENS = 400 # see ingest.py comment: 500 led to excessive truncation

@lru_cache(maxsize=1)
def get_embed_model():
    # Return the shared sentence-transformer embedding model.
    print(f"Loading embedding model: {EMBED_MODEL_NAME}")

    # NOTE: normalize_embeddings isn't applied here; callers pass at encode() time.
    return SentenceTransformer(EMBED_MODEL_NAME)

def get_engine():
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
def get_tokenizer():
    # Return the shared HuggingFace tokenizer wrapper used by the chunker.
    print(f"Loading tokenizer: {EMBED_MODEL_NAME}")

    return HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_NAME),
        max_tokens=CHUNK_MAX_TOKENS,
    )

@lru_cache(maxsize=1)
def get_chunker():
    # Return the shared HybridChunker instance.
    return HybridChunker(tokenizer=get_tokenizer(), merge_peers=True)

@lru_cache(maxsize=1)
def get_converter():
    # Return the shared Docling DocumentConverter.
 
    # NOTE: First call downloads layout models (~1-2 GB) and can take 5-15 minutes.
    #   Subsequent calls return immediately from cache.
    print("Initializing DocumentConverter (Layout models downloaded on first run, ~5-15 mins).")

    return DocumentConverter()