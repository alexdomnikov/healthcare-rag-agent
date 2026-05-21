import os
from dotenv import load_dotenv
from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.orm import Session

# All shared resources live here
from healthcare_rag.core import (
    ChunkModel,
    get_chunker,
    get_converter,
    get_embed_model,
    get_engine,
)

load_dotenv()

# project root
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data'

# Document ingestion pipeline. Parses a PDF with Docling, chunks it with 
#   HybridChunker, embeds each chunk with BGE-small, and bulk-inserts 
#   into Neon Postgres via SQLAlchemy. All heavy objects (model, engine, 
#   chunker) come from core.py so they're shared with retrieval.py and the 
#   agent tools without re-loading.

class IngestionPipeline:
    # Pipeline,  parse -> chunk -> embed -> store. Heavy objects are fetched from
    #   core.py on first use (lazy, cached).

    def __init__(self):
        self.file_path: str
        self.doc = None
        self.chunks: list[dict]

    def parse_document(self, 
                       file_path: str, 
                       debugging: bool = False):
        # Convert a PDF to a Docling Document object.

        self.file_path = file_path
        print("Parsing document.")
        self.doc = get_converter().convert(file_path).document
        print("Parsing completed.")

        if debugging:
            print("Saving structured representation to root/data/parsed.json")
            self.doc.save_as_json(DATA_PATH / "parsed.json")
            print("Saved.")

        return self.doc

    def chunk(self):
        # Produce DB-ready chunk dicts from the parsed document.
        
        if self.doc is None:
            print("Error: call parse_document() before chunk().")
            return []

        print("Generating chunks.")
        raw_chunks = list(get_chunker().chunk(dl_doc=self.doc))

        print(f"Mapping {len(raw_chunks)} chunks to DB schema.")
        chunks_for_db = []
        for chunk in raw_chunks:
            headings = chunk.meta.headings or []
            section_path = " / ".join(headings) if headings else "Unknown"

            page_num = None
            if (chunk.meta.doc_items
                    and hasattr(chunk.meta.doc_items[0], "prov")
                    and chunk.meta.doc_items[0].prov):
                page_num = chunk.meta.doc_items[0].prov[0].page_no

            chunks_for_db.append({
                "text": chunk.text,
                "page_number": page_num,
                "section_path": section_path,
                "document_source": os.path.basename(self.file_path),
                "chunk_strategy": "hybrid_chunker",
            })

        self.chunks = chunks_for_db
        print(f"Generated {len(chunks_for_db)} chunks.")
        return chunks_for_db

    def embed_and_store(self):
        # Embed all chunks and insert into Postgres.
        # NOTE: this is re-run safe. Deletes any existing rows for this
        #   (document_source, chunk_strategy) pair before inserting
        
        if not self.chunks:
            print("Error: no chunks to embed. Run chunk() first.")
            return

        print(f"Embedding {len(self.chunks)} chunks.")
        texts = [c["text"] for c in self.chunks]

        # NOTE: normalize_embeddings=True is required for pgvector cosine similarity.
        embeddings = get_embed_model().encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        print("Building ORM objects.")
        db_objects = [
            ChunkModel(
                text=chunk_data["text"],
                embedding=emb,
                page_number=chunk_data["page_number"],
                section_path=chunk_data["section_path"],
                document_source=chunk_data["document_source"],
                chunk_strategy=chunk_data["chunk_strategy"],
            )
            for chunk_data, emb in zip(self.chunks, embeddings)
        ]

        document_source = self.chunks[0]["document_source"]
        chunk_strategy = self.chunks[0]["chunk_strategy"]

        with Session(get_engine()) as session:
            deleted = session.execute(
                delete(ChunkModel).where(
                    ChunkModel.document_source == document_source,
                    ChunkModel.chunk_strategy == chunk_strategy,
                )
            )
            print(f"Deleted {deleted.rowcount} existing rows for "
                  f"'{document_source}' / '{chunk_strategy}'.")

            session.bulk_save_objects(db_objects)
            session.commit()

        print("Successfully embedded and stored all chunks.")

# Entry point
if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.parse_document(file_path = DATA_PATH / "cms_final_rule.pdf", debugging=True)
    pipeline.chunk()
    pipeline.embed_and_store()