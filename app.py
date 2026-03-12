"""
app.py
------
Streamlit web interface for the RAG Q&A system powered by Endee.

Usage:
    streamlit run app.py
"""

import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from endee import Endee
import google.generativeai as genai

# ── Config ─────────────────────────────────────────────────────────────────────
INDEX_NAME   = "rag_documents"
TOP_K        = 5
ENDEE_HOST   = "http://localhost:8080"
GEMINI_MODEL = "gemini-1.5-flash"
# ───────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DocMind — RAG Q&A with Endee",
    page_icon="🧠",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0b0f1a;
    color: #e8e8e0;
}

.main-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #64ffda, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.sub-title {
    font-size: 1rem;
    color: #8892a4;
    font-family: 'Space Mono', monospace;
    margin-bottom: 2rem;
}

.answer-box {
    background: linear-gradient(135deg, #111827, #1e2a3a);
    border: 1px solid #2d4a6e;
    border-left: 4px solid #64ffda;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    font-size: 1.05rem;
    line-height: 1.7;
}

.source-chip {
    display: inline-block;
    background: #1e2a3a;
    border: 1px solid #64ffda40;
    color: #64ffda;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.8rem;
    font-family: 'Space Mono', monospace;
    margin: 3px;
}

.chunk-box {
    background: #111827;
    border: 1px solid #2d2d3a;
    border-radius: 8px;
    padding: 1rem;
    font-size: 0.85rem;
    color: #8892a4;
    font-family: 'Space Mono', monospace;
    margin-bottom: 0.5rem;
    white-space: pre-wrap;
    word-break: break-word;
}

.metric-card {
    background: #111827;
    border: 1px solid #2d4a6e;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

.stTextInput > div > div > input {
    background-color: #111827 !important;
    color: #e8e8e0 !important;
    border: 1px solid #2d4a6e !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
}

.stButton > button {
    background: linear-gradient(135deg, #64ffda20, #a78bfa20) !important;
    color: #64ffda !important;
    border: 1px solid #64ffda !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    padding: 0.5rem 2rem !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: #64ffda20 !important;
    box-shadow: 0 0 20px #64ffda40 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🧠 DocMind</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">RAG-powered document Q&A · Endee Vector DB + Gemini</div>',
    unsafe_allow_html=True,
)
st.divider()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    gemini_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Get a free key at https://aistudio.google.com",
    )

    endee_host = st.text_input("Endee Host", value=ENDEE_HOST)
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, TOP_K)

    st.divider()
    st.markdown("### 📋 How to use")
    st.markdown("""
1. Start Endee: `docker compose up -d`
2. Add documents to `./docs/`
3. Run: `python ingest.py`
4. Ask questions here!
    """)

    st.divider()
    st.markdown("### 🔗 Architecture")
    st.markdown("""
```
User Query
    ↓
SentenceTransformer
(all-MiniLM-L6-v2)
    ↓
Endee Vector Search
(cosine similarity)
    ↓
Top-K Chunks
    ↓
Gemini LLM
    ↓
Final Answer
```
    """)


# ── Cached resources ────────────────────────────────────────────────────────────
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_endee_index(host):
    client = Endee()
    client.set_base_url(f"{host}/api/v1")
    return client.get_index(name=INDEX_NAME)


# ── Main UI ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    question = st.text_input(
        "Ask a question about your documents",
        placeholder="e.g. What is the main topic discussed in the documents?",
        label_visibility="collapsed",
    )

with col2:
    ask_btn = st.button("🔍 Ask", use_container_width=True)


# ── Suggested questions ─────────────────────────────────────────────────────────
st.markdown("**💡 Try asking:**")
suggestion_cols = st.columns(3)
suggestions = [
    "Summarise the key points",
    "What are the main conclusions?",
    "List important facts mentioned",
]
for i, (col, sug) in enumerate(zip(suggestion_cols, suggestions)):
    if col.button(sug, key=f"sug_{i}"):
        question = sug
        ask_btn = True


# ── Answer generation ───────────────────────────────────────────────────────────
if ask_btn and question:
    if not gemini_key:
        st.error("⚠️ Please enter your Gemini API Key in the sidebar.")
        st.stop()

    with st.spinner("🔍 Searching Endee vector DB …"):
        try:
            embedder = load_embedder()
            index = get_endee_index(endee_host)
            q_vec = embedder.encode(question, convert_to_list=True)
            results = index.query(vector=q_vec, top_k=top_k)
        except Exception as e:
            st.error(f"❌ Endee error: {e}\nMake sure Endee is running and documents are ingested.")
            st.stop()

    if not results:
        st.warning("No relevant documents found. Have you run `python ingest.py`?")
        st.stop()

    with st.spinner("💬 Generating answer with Gemini …"):
        try:
            genai.configure(api_key=gemini_key)
            llm = genai.GenerativeModel(GEMINI_MODEL)

            context = "\n\n---\n\n".join(
                f"[Source: {r.meta.get('source', 'unknown')}]\n{r.meta.get('text', '')}"
                for r in results
            )

            prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer isn't in the context, say "I couldn't find relevant information in the documents."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
            response = llm.generate_content(prompt)
            answer = response.text.strip()
        except Exception as e:
            st.error(f"❌ Gemini error: {e}")
            st.stop()

    # ── Display results ─────────────────────────────────────────────────────────
    st.markdown("### 📖 Answer")
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    sources = list({r.meta.get("source", "unknown") for r in results})
    st.markdown("**📎 Sources:**")
    chips = " ".join(f'<span class="source-chip">📄 {s}</span>' for s in sources)
    st.markdown(chips, unsafe_allow_html=True)

    # Metrics
    st.markdown("### 📊 Retrieval Stats")
    m1, m2, m3 = st.columns(3)
    m1.metric("Chunks Retrieved", len(results))
    m2.metric("Sources Used", len(sources))
    m3.metric("Model", "Gemini 1.5 Flash")

    # Retrieved context
    with st.expander("🔎 View retrieved chunks"):
        for i, r in enumerate(results):
            st.markdown(
                f'<div class="chunk-box"><b>Chunk {i+1}</b> · '
                f'Score: {r.similarity:.3f} · Source: {r.meta.get("source", "?")}\n\n'
                f'{r.meta.get("text", "")}</div>',
                unsafe_allow_html=True,
            )

elif ask_btn and not question:
    st.warning("Please enter a question first.")


# ── Footer ──────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<p style="text-align:center; color:#4a5568; font-size:0.8rem; font-family: monospace;">'
    'Built with Endee Vector DB · sentence-transformers · Google Gemini · Streamlit</p>',
    unsafe_allow_html=True,
)
