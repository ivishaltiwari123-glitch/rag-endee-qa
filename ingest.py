"""
ingest.py
---------
Loads documents (PDF, DOCX, TXT) from the /docs folder,
splits them into chunks, generates embeddings using
sentence-transformers, and stores them in Endee vector DB.

Usage:
    python ingest.py
"""

import os
import glob
import uuid
from pathlib import Path

from sentence_transformers import SentenceTransformer # type: ignore
from endee import Endee, Precision # type: ignore
import PyPDF2 # type: ignore
import docx # type: ignore

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_DIR       = "./docs"
INDEX_NAME     = "rag_documents"
EMBEDDING_DIM  = 384          # all-MiniLM-L6-v2 output size
CHUNK_SIZE     = 400           # characters per chunk
CHUNK_OVERLAP  = 80            # overlap between chunks
ENDEE_HOST     = "http://localhost:8080"
# ──────────────────────────────────────────────────────────────────────────────


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf(path: str) -> str:
    text = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)


def load_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def load_document(path: str) -> str:
    """Dispatch to the right loader based on file extension."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext == ".docx":
        return load_docx(path)
    else:
        return load_text_file(path)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 30]   # drop tiny remnants


def main():
    # 1. Connect to Endee
    print("🔌 Connecting to Endee vector DB …")
    client = Endee()
    client.set_base_url(f"{ENDEE_HOST}/api/v1")

    # 2. (Re)create index
    try:
        client.delete_index(INDEX_NAME)
        print(f"   Deleted existing index '{INDEX_NAME}'")
    except:
        pass

    client.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        space_type="cosine",
        precision=Precision.INT8,
    )
    print(f"   ✅ Index '{INDEX_NAME}' created (dim={EMBEDDING_DIM}, cosine, INT8)")

    index = client.get_index(name=INDEX_NAME)

    # 3. Load embedding model
    print("🤖 Loading embedding model (all-MiniLM-L6-v2) …")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 4. Walk docs directory
    patterns = ["**/*.txt", "**/*.pdf", "**/*.docx", "**/*.md"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(DOCS_DIR, pat), recursive=True))

    if not files:
        print(f"⚠️  No documents found in '{DOCS_DIR}'. Add some files and re-run.")
        return

    total_chunks = 0

    for filepath in files:
        print(f"\n📄 Processing: {filepath}")
        raw = load_document(filepath)
        chunks = chunk_text(raw)
        print(f"   → {len(chunks)} chunks")

        # Build vectors batch
        vectors = model.encode(chunks, show_progress_bar=True).tolist()

        items = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            items.append({
                "id": str(uuid.uuid4()),
                "vector": vec,
                "meta": {
                    "source": os.path.basename(filepath),
                    "chunk_index": i,
                    "text": chunk,
                },
            })

        # Upsert in batches of 256
        batch_size = 256
        for b in range(0, len(items), batch_size):
            index.upsert(items[b : b + batch_size])

        total_chunks += len(chunks)

    print(f"\n✅ Done! Indexed {total_chunks} chunks from {len(files)} file(s).")


if __name__ == "__main__":
    main()
