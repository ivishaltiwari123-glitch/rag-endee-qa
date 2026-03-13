# 💼 JobMatch AI — Resume-to-Job Matcher using Endee Vector Database

> Upload your resume → Endee semantically searches 100+ job descriptions → Groq AI explains exactly why each job matches your profile.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Endee](https://img.shields.io/badge/Vector_DB-Endee-64ffda?style=flat-square)
![Groq](https://img.shields.io/badge/LLM-Groq_LLaMA3-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Problem Statement

Job searching is broken. Candidates keyword-stuff resumes. Recruiters miss great candidates. Traditional search fails because it matches words, not meaning.

**JobMatch AI solves this** by converting both resumes and job descriptions into semantic vector embeddings stored in **Endee vector database**, then finding matches based on actual meaning and skill similarity — not just keyword overlap.

---

## 📸 Screenshots

> After running the app, take screenshots and save them in a `screenshots/` folder in this repo.

---

## 🏗️ System Design

```
┌─────────────────────────────────────────────────────────────┐
│                     INDEXING PIPELINE                        │
│                                                              │
│  100 Job Descriptions                                        │
│       │                                                      │
│       ▼                                                      │
│  Text Representation                                         │
│  (title + company + skills + description)                    │
│       │                                                      │
│       ▼                                                      │
│  SentenceTransformer (all-MiniLM-L6-v2)                     │
│  384-dimensional embeddings                                  │
│       │                                                      │
│       ▼                                                      │
│  Endee Vector DB                                             │
│  (cosine similarity, INT8 quantisation)                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     MATCHING PIPELINE                        │
│                                                              │
│  User uploads Resume (PDF / TXT)                            │
│       │                                                      │
│       ▼                                                      │
│  Text Extraction (PyPDF2)                                    │
│       │                                                      │
│       ▼                                                      │
│  SentenceTransformer → 384-dim resume vector                │
│       │                                                      │
│       ▼                                                      │
│  Endee.query(top_k=8)                                        │
│  Cosine similarity search across 100 job vectors            │
│       │                                                      │
│       ▼                                                      │
│  Ranked Job Results (with similarity scores)                │
│       │                                                      │
│       ▼                                                      │
│  Groq LLaMA3-8B                                             │
│  Personalised career analysis + skill gap advice            │
│       │                                                      │
│       ▼                                                      │
│  Final UI: Job Cards + AI Report + Match Stats              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔑 How Endee Vector Database is Used

Endee is the **core engine** of this project — not just a plugin:

| Operation | Endee API | Purpose |
|---|---|---|
| Create index | `client.create_index(name, dimension=384, space_type="cosine", precision=INT8)` | Creates optimised cosine similarity index with INT8 quantisation for fast search |
| Index jobs | `index.upsert([{id, vector, meta}])` | Stores 100 job embeddings with full metadata (title, company, skills, description) |
| Match resume | `index.query(vector=resume_vec, top_k=8)` | Finds 8 most semantically similar jobs to the resume in milliseconds |
| Metadata retrieval | `result.get('meta', {})` | Retrieves job details alongside similarity scores for display |

**Why Endee over other vector DBs?**
- INT8 quantisation reduces memory footprint without losing accuracy
- Single-node deployment handles our job index with millisecond latency
- Clean Python SDK makes integration straightforward
- Self-hosted — no API keys or rate limits for vector search

---

## ✨ Features

- 📄 **Resume PDF upload** — extract and embed resume text automatically
- ✏️ **Skills paste** — alternatively just type your skills
- 🔍 **Semantic job search** — Endee finds jobs by meaning, not keywords
- 🥇 **Ranked results** — jobs sorted by cosine similarity score
- 🏷️ **Skill tags** — see required skills at a glance per job
- 🤖 **AI career analysis** — Groq explains your fit, gaps, and top recommendation
- 📊 **Match statistics** — top match %, average score, total jobs searched
- 💼 **100 real job listings** — across Google, Microsoft, Amazon, Flipkart, Endee.io and 90+ companies

---

## 📁 Project Structure

```
rag-endee-qa/
├── docker-compose.yml     # Spins up Endee vector DB server
├── ingest_jobs.py         # Embeds & indexes 100 jobs into Endee
├── jobs_data.py           # Dataset of 100 job descriptions
├── app_jobs.py            # Streamlit web UI
└── requirements.txt       # Python dependencies
```

---

## 🚀 Setup & Running

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Free Groq API key → [console.groq.com](https://console.groq.com)

### Step 1 — Clone & fork (mandatory per evaluation)

```bash
# Star and fork https://github.com/endee-io/endee first, then:
git clone https://github.com/ivishaltiwari123-glitch/rag-endee-qa
cd rag-endee-qa
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Start Endee vector DB

```bash
docker compose up -d
```

Endee dashboard available at → **http://localhost:8080**

### Step 4 — Index 100 job descriptions into Endee

```bash
python ingest_jobs.py
```

Expected output:
```
🔌 Connecting to Endee vector DB …
   ✅ Index 'job_listings' created (dim=384, cosine, INT8)
🤖 Loading embedding model …
📋 Indexing 100 job listings …
✅ Done! Indexed 100 jobs into Endee.
```

### Step 5 — Launch the app

```bash
python -m streamlit run app_jobs.py
```

Open → **http://localhost:8501**

### Step 6 — Use the app

1. Paste your Groq API key in the sidebar
2. Upload your resume PDF **or** paste your skills
3. Click **Find Matching Jobs**
4. See ranked jobs + AI career analysis!

---

## 🧪 Example

**Input:** Resume with skills — Python, Machine Learning, Deep Learning, NLP, PyTorch

**Output:**
```
🥇 ML Engineer - NLP @ Sprinklr          — 68% match
🥈 NLP Engineer @ Sarvam AI              — 65% match
🥉 Deep Learning Researcher @ Samsung    — 63% match
#4 AI/ML Intern @ Endee.io               — 61% match
```

**AI Analysis:**
> Your profile strongly aligns with NLP and ML engineering roles.
> Top match is Sprinklr because your PyTorch and transformer experience
> directly maps to their text classification needs.
> Recommend learning LangChain to unlock more GenAI roles.
> Overall fit score: 7.5/10

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Vector Database | **Endee** (self-hosted, open-source) |
| Embeddings | **sentence-transformers** `all-MiniLM-L6-v2` (384-dim) |
| LLM for Analysis | **Groq LLaMA3-8B** (free, ultra-fast) |
| Web UI | **Streamlit** |
| PDF Parsing | **PyPDF2** |
| Containerisation | **Docker Compose** |

---

## 🔧 Configuration

| Parameter | File | Default | Description |
|---|---|---|---|
| `TOP_K` | `app_jobs.py` | 8 | Number of jobs to retrieve |
| `EMBEDDING_DIM` | `ingest_jobs.py` | 384 | Embedding size |
| `GROQ_MODEL` | `app_jobs.py` | `llama-3.1-8b-instant` | Groq LLM model |
| `ENDEE_HOST` | Both files | `http://localhost:8080` | Endee server URL |

---

## 📄 License

MIT © 2026 — Vishal Tiwari
