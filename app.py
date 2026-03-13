"""
app.py
------
Streamlit web interface for the RAG Q&A system powered by Endee.

Usage:
    streamlit run app.py
"""

import os
import streamlit as st # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
from endee import Endee # type: ignore
from groq import Groq # type: ignore

# ── Config ─────────────────────────────────────────────────────────────────────
INDEX_NAME   = "rag_documents"
TOP_K        = 5
ENDEE_HOST   = "http://localhost:8080"
GROQ_MODEL   = "llama-3.1-8b-instant"
# ───────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DocMind — RAG Q&A with Endee",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; background-color: #0b0f1a; color: #e8e8e0; }
.main-title { font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, #64ffda, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0; }
.sub-title { font-size: 1rem; color: #8892a4; font-family: 'Space Mono', monospace; margin-bottom: 2rem; }
.answer-box { background: linear-gradient(135deg, #111827, #1e2a3a); border: 1px solid #2d4a6e; border-left: 4px solid #64ffda; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; font-size: 1.05rem; line-height: 1.7; }
.source-chip { display: inline-block; background: #1e2a3a; border: 1px solid #64ffda40; color: #64ffda; border-radius: 20px; padding: 2px 12px; font-size: 0.8rem; font-family: 'Space Mono', monospace; margin: 3px; }
.chunk-box { background: #111827; border: 1px solid #2d2d3a; border-radius: 8px; padding: 1rem; font-size: 0.85rem; color: #8892a4; font-family: 'Space Mono', monospace; margin-bottom: 0.5rem; white-space: pre-wrap; word-break: break-word; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧠 DocMind</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">RAG-powered document Q&A · Endee Vector DB + Groq LLaMA3</div>', unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    groq_key = st.text_input("Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""), help="Get a free key at https://console.groq.com")
    endee_host = st.text_input("Endee Host", value=ENDEE_HOST)
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, TOP_K)
    st.divider()
    st.markdown("### 📋 How to use")
    st.markdown("1. Start Endee: `docker compose up -d`\n2. Add documents to `./docs/`\n3. Run: `python ingest.py`\n4. Ask questions here!")
    st.divider()
    st.markdown("### 🔗 Architecture")
    st.markdown("```\nUser Query\n    ↓\nSentenceTransformer\n    ↓\nEndee Vector Search\n    ↓\nTop-K Chunks\n    ↓\nGroq LLaMA3\n    ↓\nFinal Answer\n```")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_endee_index(host):
    client = Endee()
    client.set_base_url(f"{host}/api/v1")
    return client.get_index(name=INDEX_NAME)

col1, col2 = st.columns([3, 1])
with col1:
    question = st.text_input("Ask a question", placeholder="e.g. What is the main topic discussed in the documents?", label_visibility="collapsed")
with col2:
    ask_btn = st.button("🔍 Ask", use_container_width=True)

st.markdown("**💡 Try asking:**")
suggestion_cols = st.columns(3)
suggestions = ["Summarise the key points", "What are the main conclusions?", "List important facts mentioned"]
for i, (col, sug) in enumerate(zip(suggestion_cols, suggestions)):
    if col.button(sug, key=f"sug_{i}"):
        question = sug
        ask_btn = True

if ask_btn and question:
    if not groq_key:
        st.error("⚠️ Please enter your Groq API Key in the sidebar.")
        st.stop()

    with st.spinner("🔍 Searching Endee vector DB …"):
        try:
            embedder = load_embedder()
            index = get_endee_index(endee_host)
            q_vec = embedder.encode(question).tolist()
            results = index.query(vector=q_vec, top_k=top_k)
        except Exception as e:
            st.error(f"❌ Endee error: {e}")
            st.stop()

    if not results:
        st.warning("No relevant documents found. Have you run `python ingest.py`?")
        st.stop()

    with st.spinner("💬 Generating answer with Groq LLaMA3 …"):
        try:
            context = "\n\n---\n\n".join(
                f"[Source: {r.get('meta', {}).get('source', 'unknown')}]\n{r.get('meta', {}).get('text', '')}"
                for r in results
            )
            prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer isn't in the context, say "I couldn't find relevant information in the documents."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
            groq_client = Groq(api_key=groq_key)
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"❌ Groq error: {e}")
            st.stop()

    st.markdown("### 📖 Answer")
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    sources = list({r.get('meta', {}).get("source", "unknown") for r in results})
    st.markdown("**📎 Sources:**")
    chips = " ".join(f'<span class="source-chip">📄 {s}</span>' for s in sources)
    st.markdown(chips, unsafe_allow_html=True)

    st.markdown("### 📊 Retrieval Stats")
    m1, m2, m3 = st.columns(3)
    m1.metric("Chunks Retrieved", len(results))
    m2.metric("Sources Used", len(sources))
    m3.metric("Model", "LLaMA3-8B")

    with st.expander("🔎 View retrieved chunks"):
        for i, r in enumerate(results):
            st.markdown(
                f'<div class="chunk-box"><b>Chunk {i+1}</b> · Source: {r.get("meta", {}).get("source", "?")}\n\n{r.get("meta", {}).get("text", "")}</div>',
                unsafe_allow_html=True,
            )

elif ask_btn and not question:
    st.warning("Please enter a question first.")

st.divider()
st.markdown('<p style="text-align:center; color:#4a5568; font-size:0.8rem; font-family: monospace;">Built with Endee Vector DB · sentence-transformers · Groq LLaMA3 · Streamlit</p>', unsafe_allow_html=True)
