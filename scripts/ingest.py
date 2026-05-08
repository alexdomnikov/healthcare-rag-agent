import os
from dotenv import load_dotenv

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, Column, Integer, Text, delete
from sqlalchemy.orm import declarative_base, Session
from pgvector.sqlalchemy import Vector

load_dotenv()
Base = declarative_base()

# This tells SQLAlchemy what the Postgres table looks like
class ChunkModel(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True)
    text = Column(Text, nullable=False)
    # 384 matches BGE-small's output
    embedding = Column(Vector(384))
    page_number = Column(Integer)
    section_path = Column(Text)
    document_source = Column(Text, nullable=False)
    chunk_strategy = Column(Text, nullable=False)

class IngestionPipeline:
    def __init__(self):
        # Initializing models once
        print('Initializing models. This may take a few moments.')
        self.converter = DocumentConverter()
        self.embed_model = SentenceTransformer('BAAI/bge-small-en-v1.5')

        # Initialize the chunker with the BGE tokenizer
        print('Initializing tokenizer & chunker.')
        self.tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5"),
            max_tokens=500,
        )
        self.chunker = HybridChunker(tokenizer=self.tokenizer, merge_peers=True)

        # Variable for parsed document & stored chunks
        self.doc = None
        self.chunks = None

        db_url = os.environ.get('DATABASE_URL')
        if not db_url:
            raise ValueError("DATABASE_URL environment variable is not set.")
        # Neon hands out 'postgresql://' URLs; SQLAlchemy maps that to psycopg2 by default.
        # I installed psycopg v3, so I'm rewriting the scheme to point SQLAlchemy at the right driver.
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
        self.engine = create_engine(db_url)

    def parse_document(self, file_path: str, debugging=False):
        print('Parsing document. First run downloads layout models (~5-15 min); subsequent runs are faster.')
        self.doc = self.converter.convert(file_path).document
        print('Parsing completed.')

        # Export structured representation for debugging
        if debugging:
            print('Saving to a .json file.')
            self.doc.save_as_json("../data/parsed.json")
            print('File saved as ../data/parsed.json')

        return self.doc

    def chunk(self):
        if self.doc is None:
            print("Error: no document saved for chunking.")
            return []

        # Generate the raw Chunk objects
        print("Generating Chunk objects")
        raw_chunks = self.chunker.chunk(dl_doc=self.doc)

        # Map Docling's objects to the Postgres schema
        print("Mapping chunks to Postgres schema")
        chunks_for_db = []
        for chunk in raw_chunks:
            # Extract headings (docling provides them as a list of strings)
            headings = chunk.meta.headings
            section_path = " / ".join(headings) if headings else "Unknown"

            # Extract page numbers (we pull from the first document item in the chunk)
            page_num = None
            if chunk.meta.doc_items and hasattr(chunk.meta.doc_items[0], "prov") and chunk.meta.doc_items[0].prov:
                page_num = chunk.meta.doc_items[0].prov[0].page_no

            # Package it for SQLAlchemy
            chunks_for_db.append({
                "text": chunk.text,
                "page_number": page_num,
                "section_path": section_path,
                "document_source": "cms_final_rule.pdf",
                "chunk_strategy": "hybrid_chunker"
            })

        self.chunks = chunks_for_db
        print(f"Generated {len(chunks_for_db)} chunks.")
        return chunks_for_db

    def embed_and_store(self):
        if not self.chunks:
            print("Error: No chunks available to embed.")
            return []

        print(f"Embedding {len(self.chunks)} chunks.")

        # Extract texts and generate vectors
        texts = [chunk["text"] for chunk in self.chunks]

        # NOTE: normalize_embeddings=True is required for pgvector cosine similarity
        embeddings = self.embed_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        # Combine the dictionaries and embeddings into SQLAlchemy ORM objects
        print("Formatting ORM objects.")
        db_objects = []
        for chunk_data, emb in zip(self.chunks, embeddings):
            db_objects.append(
                ChunkModel(
                    text=chunk_data["text"],
                    embedding=emb,
                    page_number=chunk_data["page_number"],
                    section_path=chunk_data["section_path"],
                    document_source=chunk_data["document_source"],
                    chunk_strategy=chunk_data["chunk_strategy"]
                )
            )

        # Re-run safety: delete existing rows for this (document_source, chunk_strategy)
        #   combination before inserting, in the same transaction. Lets me re-run ingestion
        #   without duplicates.
        document_source = self.chunks[0]["document_source"]
        chunk_strategy = self.chunks[0]["chunk_strategy"]
        print(f"Clearing existing rows where document_source='{document_source}' and chunk_strategy='{chunk_strategy}'.")
        with Session(self.engine) as session:
            deleted = session.execute(
                delete(ChunkModel).where(
                    ChunkModel.document_source == document_source,
                    ChunkModel.chunk_strategy == chunk_strategy,
                )
            )
            print(f"Deleted {deleted.rowcount} existing rows.")
            print("Executing bulk save to Neon Postgres.")
            session.bulk_save_objects(db_objects)
            session.commit()

        print("Successfully embedded and stored all chunks.")

if __name__ == '__main__':
    pipeline = IngestionPipeline()
    pipeline.parse_document("../data/cms_final_rule.pdf")
    pipeline.chunk()
    pipeline.embed_and_store()