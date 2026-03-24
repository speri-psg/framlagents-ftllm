# ingest.py — Run once to build the local knowledge base
# Usage: python ingest.py
# Re-run whenever you add/update documents in the docs/ folder

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import pypdf
import pandas as pd
import os
import glob

DOCS_FOLDER = "docs"
CHROMA_PATH = "chroma_db"
CHUNK_SIZE = 500    # words per chunk
CHUNK_OVERLAP = 50  # words overlap between chunks


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def ingest_pdf(path):
    reader = pypdf.PdfReader(path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)
    full_text = " ".join(pages_text)
    return chunk_text(full_text)


def ingest_csv(path):
    df = pd.read_csv(path)
    chunks = [df[i:i + 30].to_string() for i in range(0, len(df), 30)]
    return [c for c in chunks if c.strip()]


def main():
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Fresh ingest — drop existing collection
    try:
        client.delete_collection("framl_kb")
        print("Dropped existing collection for fresh ingest.")
    except Exception:
        pass

    collection = client.create_collection("framl_kb", embedding_function=ef)

    all_chunks, all_ids, all_metadata = [], [], []
    chunk_id = 0

    # Ingest PDFs from docs/
    pdf_files = glob.glob(os.path.join(DOCS_FOLDER, "*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {DOCS_FOLDER}/  — add your documents there and re-run.")
    for pdf_path in pdf_files:
        fname = os.path.basename(pdf_path)
        print(f"Ingesting PDF: {fname} ...")
        try:
            chunks = ingest_pdf(pdf_path)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_ids.append(f"chunk_{chunk_id}")
                all_metadata.append({"source": fname, "type": "pdf"})
                chunk_id += 1
            print(f"  -> {len(chunks)} chunks")
        except Exception as e:
            print(f"  ERROR reading {fname}: {e}")

    # CSVs are NOT ingested into ChromaDB — data analysis is done directly
    # via pandas in the app. ChromaDB is reserved for policy/guideline PDFs.

    if not all_chunks:
        print("\nNo documents ingested. Add PDFs/CSVs to the docs/ folder and re-run.")
        return

    # Batch insert into ChromaDB
    BATCH = 100
    print(f"\nEmbedding and storing {chunk_id} chunks ...")
    for i in range(0, len(all_chunks), BATCH):
        collection.add(
            documents=all_chunks[i:i + BATCH],
            ids=all_ids[i:i + BATCH],
            metadatas=all_metadata[i:i + BATCH]
        )
        print(f"  Stored {min(i + BATCH, chunk_id)}/{chunk_id} chunks", end="\r")

    print(f"\nDone. {chunk_id} chunks stored in {CHROMA_PATH}/")
    print("You can now start the app: python application.py")


if __name__ == "__main__":
    main()