# 🧠 DocMind — RAG-powered Document Q&A using Endee Vector Database

> Ask questions about your documents in plain English. DocMind retrieves semantically relevant content using **Endee**, then generates precise answers using **Google Gemini**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Endee](https://img.shields.io/badge/Vector_DB-Endee-64ffda?style=flat-square)
![Gemini](https://img.shields.io/badge/LLM-Gemini_1.5_Flash-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Problem Statement

Enterprise knowledge is scattered across PDFs, Word documents, and text files. Traditional keyword search misses the meaning behind questions. This project solves that by building a **semantic RAG pipeline** — documents are indexed as vector embeddings in Endee, enabling meaning-based retrieval before an LLM synthesises the final answer.

---

## 🏗️ System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                        │
│                                                                   │
│  PDF / DOCX / TXT ──► Text Extraction ──► Chunking (400 chars)  │
│                                               │                   │
│                              SentenceTransformer (384-dim)        │
│                              all-MiniLM-L6-v2                    │
│                                               │                   │
│                              Endee Vector DB  │                   │
│                              (cosine, INT8)  ◄┘                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE                           │
│                                                                   │
│  User Question                                                    │
│       │                                                           │
│       ▼                                                           │
│  SentenceTransformer ──► 384-dim query vector                    │
│       │                                                           │
│       ▼                                                           │
│  Endee.query(top_k=5) ──► Top-5 similar chunks + scores         │
│       │                                                           │
│       ▼                                                           │
│  Prompt construction: [context chunks] + [question]             │
│       │                                                           │
│       ▼                                                           │
│  Google Gemini 1.5 Flash ──► Final Answer                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔑 How Endee is Used

Endee is the **core vector storage and retrieval engine** in this project:

| Operation | Endee API Used | Purpose |
|---|---|---|
| Index creation | `client.create_index(name, dimension=384, space_type="cosine", precision=INT8)` | Creates a cosine-similarity index optimised with INT8 quantisation |
| Upserting vectors | `index.upsert([{id, vector, meta}])` | Stores document chunk embeddings with metadata (source, text) |
| Semantic search | `index.query(vector, top_k=5)` | Retrieves the 5 most semantically similar chunks to the query |
| Index management | `client.list_indexes()`, `client.delete_index()` | Used during re-ingestion |

The `meta` field stores the original chunk text and source filename, so retrieved vectors carry the full context needed to generate an answer — no secondary database required.

---

## ✨ Features

- 📄 **Multi-format ingestion** — supports PDF, DOCX, TXT, and Markdown
- 🔍 **Semantic search** — vector similarity via Endee (not keyword matching)
- 💬 **LLM-powered answers** — Gemini 1.5 Flash generates grounded responses
- 🌐 **Web UI** — clean Streamlit interface with source attribution
- 🖥️ **CLI mode** — `python query.py` for terminal-based Q&A
- ⚡ **Fast** — INT8 quantisation in Endee reduces memory and speeds up search
- 🔒 **Grounded** — LLM is explicitly instructed to answer only from retrieved context

---

## 📁 Project Structure

```
rag-endee-qa/
├── docker-compose.yml   # Spins up Endee server
├── ingest.py            # Document loading, chunking, embedding & indexing
├── query.py             # CLI RAG Q&A interface
├── app.py               # Streamlit web UI
├── requirements.txt
├── docs/                # ← Put your documents here
│   └── sample.txt
└── README.md
```

---

## 🚀 Setup & Execution

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Free Gemini API key → [aistudio.google.com](https://aistudio.google.com)

---

### Step 1 — Clone & fork (mandatory per evaluation requirements)

```bash
# Fork https://github.com/endee-io/endee on GitHub first, then:
git clone https://github.com/<your-username>/endee
cd endee

# Then clone this project
git clone https://github.com/<your-username>/rag-endee-qa
cd rag-endee-qa
```

### Step 2 — Start Endee

```bash
docker compose up -d
# Endee dashboard → http://localhost:8080
```

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Add your documents

```bash
mkdir docs
cp /path/to/your/files/*.pdf docs/
# Supports: .pdf  .docx  .txt  .md
```

### Step 5 — Ingest documents into Endee

```bash
python ingest.py
```

Expected output:
```
🔌 Connecting to Endee vector DB …
   ✅ Index 'rag_documents' created (dim=384, cosine, INT8)
🤖 Loading embedding model (all-MiniLM-L6-v2) …
📄 Processing: docs/company_policy.pdf
   → 42 chunks
✅ Done! Indexed 42 chunks from 1 file(s).
```

### Step 6 — Ask questions

**Option A: Web UI**
```bash
export GEMINI_API_KEY=your_key_here
streamlit run app.py
# Open http://localhost:8501
```

**Option B: Command line**
```bash
export GEMINI_API_KEY=your_key_here
python query.py --question "What is the refund policy?"
```

---

## 🔧 Configuration

| Parameter | File | Default | Description |
|---|---|---|---|
| `CHUNK_SIZE` | `ingest.py` | 400 | Characters per text chunk |
| `CHUNK_OVERLAP` | `ingest.py` | 80 | Overlap between chunks |
| `TOP_K` | `query.py` / `app.py` | 5 | Number of chunks to retrieve |
| `EMBEDDING_DIM` | `ingest.py` | 384 | Embedding dimension (fixed to model) |
| `ENDEE_HOST` | All files | `http://localhost:8080` | Endee server URL |

---

## 🧪 Example

**Question:** *"What are the key responsibilities of a software engineer?"*

**Retrieved chunks:** 5 relevant passages from uploaded job descriptions

**Answer:**
> Based on the documents, a software engineer is responsible for designing and implementing software features, conducting code reviews, collaborating with cross-functional teams, and maintaining system reliability. They are also expected to write unit tests and participate in architectural discussions.
>
> *Sources: job_description.pdf*

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Vector Database | **Endee** (open-source, self-hosted) |
| Embeddings | **sentence-transformers** `all-MiniLM-L6-v2` |
| LLM | **Google Gemini 1.5 Flash** |
| Web UI | **Streamlit** |
| PDF parsing | **PyPDF2** |
| DOCX parsing | **python-docx** |
| Containerisation | **Docker Compose** |

---

## 📄 License

MIT © 2026
