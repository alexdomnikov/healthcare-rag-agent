from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import text
from docling.document_converter import DocumentConverter
from transformers import AutoTokenizer

from healthcare_rag.core import get_embed_model, get_engine, ChunkModel, EMBED_MODEL_NAME

load_dotenv()

# project root
ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / 'data' / "cms_final_rule.pdf"
# BGE-small has a 512-token limit; I was forced to use CHUNK_TOKENS = 400
#   in my hybrid chunker because setting the limit to 500 output 600-700+
#   token chunks as Docling uses the chunk limit as a soft limit for
#   structure-aware chunking.
CHUNK_TOKENS = 400
OVERLAP_TOKENS = 50
BATCH_SIZE = 32
DOC_SOURCE = "cms_final_rule.pdf"
STRATEGY = "fixed"

# Re-ingest the CMS PDF with naive fixed-size token chunking (512 tokens,
#   50-token overlap) and insert into the chunks table under chunk_strategy='fixed'.

# Run: uv run scripts/ingest_fixed_chunks.py
# Run once before chunking_strategy.py. 
# NOTE: Safe to re-run, existing 'fixed' chunks are deleted first.

def parse_doc():
    # Parse the PDF with Docling. Results are cached after first run.
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")
    print(f"Parsing {PDF_PATH}")
    result = DocumentConverter().convert(str(PDF_PATH))
    return result.document


def extract_items(doc) -> list[dict]:
    # Walk the Docling document tree; return text items with page numbers.
    items = []
    for item, _level in doc.iterate_items():
        text_val = getattr(item, "text", None) or getattr(item, "content", None)
        if not text_val or not text_val.strip():
            continue

        page = None
        prov = getattr(item, "prov", None)
        if prov:
            entry = prov[0] if isinstance(prov, list) else prov
            page = getattr(entry, "page_no", None)

        items.append({"text": text_val.strip(), "page": page})

    return items


def chunk_fixed(items: list[dict]) -> list[dict]:
    # Concatenate all text into one token stream, slide a 512-token window
    #   with 50-token overlap, assign each chunk the median page of its window.
   
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)

    all_tokens: list[int] = []
    token_pages: list[int | None] = []

    for item in items:
        toks = tokenizer.encode(item["text"], add_special_tokens=False)
        all_tokens.extend(toks)
        token_pages.extend([item["page"]] * len(toks))

    chunks = []
    i = 0
    while i < len(all_tokens):
        window_toks = all_tokens[i : i + CHUNK_TOKENS]
        window_pages = [p for p in token_pages[i : i + CHUNK_TOKENS] if p is not None]

        chunk_text = tokenizer.decode(window_toks, skip_special_tokens=True).strip()
        page_num = int(np.median(window_pages)) if window_pages else None

        if chunk_text:
            chunks.append({"text": chunk_text, "page_number": page_num})

        i += CHUNK_TOKENS - OVERLAP_TOKENS

    return chunks

def embed_and_insert(chunks: list[dict]) -> None:
    model = get_embed_model()
    engine = get_engine()

    print(f"Embedding {len(chunks)} chunks (batch_size={BATCH_SIZE})")
    embeddings = model.encode(
        [c["text"] for c in chunks],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    with Session(engine) as session:
        deleted = session.execute(
            text("DELETE FROM chunks WHERE chunk_strategy = :s"),
            {"s": STRATEGY},
        ).rowcount
        if deleted:
            print(f"Deleted {deleted} existing '{STRATEGY}' chunks.")

        objects = [
            ChunkModel(
                text=c["text"],
                embedding=emb.tolist(),
                page_number=c["page_number"],
                section_path=None,
                document_source=DOC_SOURCE,
                chunk_strategy=STRATEGY,
            )
            for c, emb in zip(chunks, embeddings)
        ]
        session.bulk_save_objects(objects)
        session.commit()

    print(f"Inserted {len(objects)} fixed-size chunks (chunk_strategy='{STRATEGY}').")

def main():
    doc = parse_doc()
    items = extract_items(doc)
    print(f"Extracted {len(items)} text items from document.")

    chunks = chunk_fixed(items)
    print(f"Produced {len(chunks)} chunks ({CHUNK_TOKENS}-token window, {OVERLAP_TOKENS}-token overlap).")

    embed_and_insert(chunks)

    # Sanity check
    engine = get_engine()
    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT count(*), round(avg(length(text))) FROM chunks WHERE chunk_strategy='fixed'")
        ).fetchone()
    print(f"\nDB sanity: {count[0]} rows, avg text length {count[1]} chars")

if __name__ == "__main__":
    main()