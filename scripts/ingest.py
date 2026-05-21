import argparse
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import delete
from sqlalchemy.orm import Session
from transformers import AutoTokenizer

from healthcare_rag.core import (
    CHUNK_MAX_TOKENS,
    EMBED_MODEL_NAME,
    ChunkModel,
    get_chunker,
    get_converter,
    get_embed_model,
    get_engine,
)

# Parse the CMS PDF, chunk it, embed each chunk with BGE-small, and bulk-insert
# into Postgres. Two chunking strategies:
#
#   --strategy hybrid  (default) Docling HybridChunker, structure-aware. Stored
#                      under chunk_strategy='hybrid_chunker', section_path filled in.
#   --strategy fixed   naive fixed-size token windows with overlap. Stored under
#                      chunk_strategy='fixed', section_path NULL. Needed for
#                      eval/ablations/chunking_strategy.py.
#
# Safe to re-run: rows for the same (document_source, chunk_strategy) are
# deleted before insert.

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data"
DEFAULT_PDF = DATA_PATH / "cms_final_rule.pdf"

FIXED_WINDOW_TOKENS = CHUNK_MAX_TOKENS
FIXED_OVERLAP_TOKENS = 50
BATCH_SIZE = 32


def parse_pdf(pdf_path: Path, save_debug_json: bool = False):
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")
    print(f"Parsing {pdf_path}")
    doc = get_converter().convert(str(pdf_path)).document
    if save_debug_json:
        out = DATA_PATH / "parsed.json"
        doc.save_as_json(out)
        print(f"Saved structured representation to {out}")
    return doc


def chunks_hybrid(doc, source: str) -> list[dict]:
    raw = list(get_chunker().chunk(dl_doc=doc))
    print(f"Mapping {len(raw)} chunks to DB schema.")
    out = []
    for chunk in raw:
        headings = chunk.meta.headings or []
        section_path = " / ".join(headings) if headings else "Unknown"

        page_num = None
        if (chunk.meta.doc_items
                and hasattr(chunk.meta.doc_items[0], "prov")
                and chunk.meta.doc_items[0].prov):
            page_num = chunk.meta.doc_items[0].prov[0].page_no

        out.append({
            "text": chunk.text,
            "page_number": page_num,
            "section_path": section_path,
            "document_source": source,
            "chunk_strategy": "hybrid_chunker",
        })
    return out


def chunks_fixed(doc, source: str) -> list[dict]:
    # Flatten the document to a list of (text, page_number) items, encode the
    # whole stream once, then slide a window over the token IDs. Each chunk's
    # page is the median of its tokens' source pages.
    items: list[tuple[str, int | None]] = []
    for item, _level in doc.iterate_items():
        text_val = getattr(item, "text", None) or getattr(item, "content", None)
        if not text_val or not text_val.strip():
            continue
        page = None
        prov = getattr(item, "prov", None)
        if prov:
            entry = prov[0] if isinstance(prov, list) else prov
            page = getattr(entry, "page_no", None)
        items.append((text_val.strip(), page))
    print(f"Extracted {len(items)} text items.")

    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
    all_tokens: list[int] = []
    token_pages: list[int | None] = []
    for text_val, page in items:
        toks = tokenizer.encode(text_val, add_special_tokens=False)
        all_tokens.extend(toks)
        token_pages.extend([page] * len(toks))

    out: list[dict] = []
    step = FIXED_WINDOW_TOKENS - FIXED_OVERLAP_TOKENS
    i = 0
    while i < len(all_tokens):
        window_toks = all_tokens[i:i + FIXED_WINDOW_TOKENS]
        window_pages = [p for p in token_pages[i:i + FIXED_WINDOW_TOKENS] if p is not None]
        chunk_text = tokenizer.decode(window_toks, skip_special_tokens=True).strip()
        if chunk_text:
            page_num = int(np.median(window_pages)) if window_pages else None
            out.append({
                "text": chunk_text,
                "page_number": page_num,
                "section_path": None,
                "document_source": source,
                "chunk_strategy": "fixed",
            })
        i += step

    print(f"Produced {len(out)} chunks ({FIXED_WINDOW_TOKENS}-token window, "
          f"{FIXED_OVERLAP_TOKENS}-token overlap).")
    return out


def embed_and_store(chunks: list[dict]) -> None:
    if not chunks:
        print("No chunks to embed.")
        return

    print(f"Embedding {len(chunks)} chunks.")
    # normalize_embeddings=True is required for pgvector cosine similarity.
    embeddings = get_embed_model().encode(
        [c["text"] for c in chunks],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    db_objects = [
        ChunkModel(
            text=c["text"],
            embedding=emb if not hasattr(emb, "tolist") else emb.tolist(),
            page_number=c["page_number"],
            section_path=c["section_path"],
            document_source=c["document_source"],
            chunk_strategy=c["chunk_strategy"],
        )
        for c, emb in zip(chunks, embeddings)
    ]

    source = chunks[0]["document_source"]
    strategy = chunks[0]["chunk_strategy"]

    with Session(get_engine()) as session:
        deleted = session.execute(
            delete(ChunkModel).where(
                ChunkModel.document_source == source,
                ChunkModel.chunk_strategy == strategy,
            )
        )
        print(f"Deleted {deleted.rowcount} existing rows for "
              f"'{source}' / '{strategy}'.")
        session.bulk_save_objects(db_objects)
        session.commit()

    print(f"Inserted {len(db_objects)} chunks (chunk_strategy='{strategy}').")


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest a PDF into the chunks table.")
    p.add_argument("--strategy", choices=["hybrid", "fixed"], default="hybrid",
                   help="Chunking strategy (default: hybrid)")
    p.add_argument("--pdf", type=Path, default=DEFAULT_PDF,
                   help=f"Path to PDF (default: {DEFAULT_PDF})")
    p.add_argument("--debug-json", action="store_true",
                   help="Also write Docling's structured JSON to data/parsed.json")
    args = p.parse_args()

    doc = parse_pdf(args.pdf, save_debug_json=args.debug_json)
    source = os.path.basename(args.pdf)

    if args.strategy == "hybrid":
        chunks = chunks_hybrid(doc, source)
    else:
        chunks = chunks_fixed(doc, source)

    embed_and_store(chunks)


if __name__ == "__main__":
    main()
