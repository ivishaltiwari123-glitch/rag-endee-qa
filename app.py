"""
app_jobs.py
-----------
Upload your resume (PDF/TXT) → get matched jobs from Endee vector DB.

Usage:
    streamlit run app_jobs.py
"""

import os
import io
import streamlit as st # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
from endee import Endee # type: ignore
from groq import Groq # type: ignore
import PyPDF2 # type: ignore

INDEX_NAME  = "job_listings"
ENDEE_HOST  = "http://localhost:8080"
GROQ_MODEL  = "llama-3.1-8b-instant"
TOP_K       = 8

st.set_page_config(page_title="JobMatch AI — Resume to Jobs", page_icon="💼", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Cabinet+Grotesk:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Cabinet Grotesk', sans-serif;
    background: #0a0a0f;
    color: #f0f0f0;
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f9a825, #ff6f00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    color: #888;
    font-family: 'DM Mono', monospace;
    font-size: 0.9rem;
    margin-bottom: 2rem;
}

.job-card {
    background: linear-gradient(135deg, #111118, #1a1a2e);
    border: 1px solid #2a2a4a;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}

.job-card:hover { border-color: #f9a825; }

.job-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f9a825;
    margin-bottom: 4px;
}

.job-company {
    font-size: 0.9rem;
    color: #aaa;
    font-family: 'DM Mono', monospace;
}

.match-score {
    display: inline-block;
    background: linear-gradient(135deg, #f9a82520, #ff6f0020);
    border: 1px solid #f9a82560;
    color: #f9a825;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.8rem;
    font-family: 'DM Mono', monospace;
    float: right;
}

.skill-tag {
    display: inline-block;
    background: #1e1e3a;
    border: 1px solid #3a3a6a;
    color: #a0a0ff;
    border-radius: 6px;
    padding: 1px 8px;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    margin: 2px;
}

.resume-box {
    background: #111118;
    border: 2px dashed #2a2a4a;
    border-radius: 12px;
    padding: 1.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #666;
    max-height: 200px;
    overflow-y: auto;
    white-space: pre-wrap;
}

.analysis-box {
    background: linear-gradient(135deg, #111118, #1a1a2e);
    border: 1px solid #f9a82540;
    border-left: 4px solid #f9a825;
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1.5rem;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">💼 JobMatch AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload your resume · Endee finds your best matching jobs · Groq explains why</div>', unsafe_allow_html=True)
st.divider()

# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    groq_key = st.text_input("Groq API Key", type="password",
                              value=os.environ.get("GROQ_API_KEY", ""),
                              help="Free key at https://console.groq.com")
    top_k = st.slider("Jobs to retrieve", 3, 15, TOP_K)
    endee_host = st.text_input("Endee Host", value=ENDEE_HOST)

    st.divider()
    st.markdown("### 🔗 How it works")
    st.markdown("""
```
Your Resume (PDF/TXT)
       ↓
Text Extraction
       ↓
SentenceTransformer
(384-dim embedding)
       ↓
Endee Vector Search
(cosine similarity)
       ↓
Top Matching Jobs
       ↓
Groq LLaMA3 Analysis
       ↓
Personalised Report
```
    """)
    st.divider()
    st.markdown("### 📊 Database")
    st.markdown("**100 jobs** indexed across:\nGoogle, Microsoft, Amazon, Flipkart, Zepto, Endee.io and 90+ more companies")


# ── Cached resources ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_index(host):
    client = Endee()
    client.set_base_url(f"{host}/api/v1")
    return client.get_index(name=INDEX_NAME)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


# ── Upload Section ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📄 Upload Your Resume")
    uploaded = st.file_uploader("Drop your resume here", type=["pdf", "txt"],
                                 label_visibility="collapsed")

    if uploaded:
        if uploaded.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded.read())
        else:
            resume_text = extract_text_from_txt(uploaded.read())

        st.success(f"✅ Resume loaded — {len(resume_text)} characters")
        st.markdown('<div class="resume-box">' + resume_text[:1500] + ("…" if len(resume_text) > 1500 else "") + '</div>', unsafe_allow_html=True)
    else:
        st.info("👆 Upload a PDF or TXT resume to get started")
        resume_text = ""

with col2:
    st.markdown("### ✏️ Or paste your resume / skills")
    manual_text = st.text_area(
        "Paste resume text or just list your skills",
        placeholder="E.g.\nPython, Machine Learning, Deep Learning, PyTorch\nWorked on NLP projects, text classification, transformer models...",
        height=200,
        label_visibility="collapsed"
    )

# Use uploaded or manual text
final_resume = resume_text.strip() or manual_text.strip()

# ── Match Button ──────────────────────────────────────────────────────────────────
st.divider()
match_btn = st.button("🔍 Find Matching Jobs", use_container_width=True, type="primary")

# ── Results ───────────────────────────────────────────────────────────────────────
if match_btn:
    if not final_resume:
        st.error("⚠️ Please upload a resume or paste your skills first.")
        st.stop()
    if not groq_key:
        st.error("⚠️ Please enter your Groq API Key in the sidebar.")
        st.stop()

    with st.spinner("🔍 Searching 100 jobs in Endee vector DB …"):
        try:
            embedder = load_embedder()
            index = get_index(endee_host)
            resume_vec = embedder.encode(final_resume).tolist()
            results = index.query(vector=resume_vec, top_k=top_k)
        except Exception as e:
            st.error(f"❌ Endee error: {e}\nMake sure Endee is running and you ran ingest_jobs.py")
            st.stop()

    if not results:
        st.warning("No matches found. Have you run `python ingest_jobs.py`?")
        st.stop()

    # ── AI Analysis ──────────────────────────────────────────────────────────────
    with st.spinner("🤖 Groq is analysing your profile …"):
        try:
            top3 = results[:3]
            jobs_summary = "\n\n".join(
                f"Job {i+1}: {r.get('meta',{}).get('title','')} at {r.get('meta',{}).get('company','')}\n"
                f"Skills required: {r.get('meta',{}).get('skills','')}\n"
                f"Description: {r.get('meta',{}).get('description','')}"
                for i, r in enumerate(top3)
            )

            prompt = f"""You are a career advisor. A candidate has uploaded their resume/skills profile.
Based on their profile and the top 3 matching jobs found via semantic search, provide:

1. A 2-line summary of the candidate's profile
2. Why the top job is a strong match (2-3 sentences)
3. One skill they should learn to improve their chances
4. Overall fit score out of 10

Be specific, encouraging, and practical. Keep it under 150 words.

CANDIDATE PROFILE:
{final_resume[:1000]}

TOP 3 MATCHING JOBS:
{jobs_summary}

YOUR ANALYSIS:"""

            groq_client = Groq(api_key=groq_key)
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            analysis = response.choices[0].message.content.strip()
        except Exception as e:
            analysis = f"Could not generate analysis: {e}"

    # ── Display Analysis ──────────────────────────────────────────────────────────
    st.markdown("### 🤖 AI Career Analysis")
    st.markdown(f'<div class="analysis-box">{analysis}</div>', unsafe_allow_html=True)

    # ── Display Job Cards ─────────────────────────────────────────────────────────
    st.markdown(f"### 🎯 Top {len(results)} Matching Jobs")
    st.caption("Ranked by semantic similarity using Endee vector search")

    for i, r in enumerate(results):
        meta = r.get("meta", {})
        score = r.get("similarity", 0)
        score_pct = int(score * 100)
        skills = meta.get("skills", "").split(", ")

        skill_tags = " ".join(f'<span class="skill-tag">{s}</span>' for s in skills[:5])
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"

        st.markdown(f"""
<div class="job-card">
    <span class="match-score">{'⭐ ' if i < 3 else ''}{score_pct}% match</span>
    <div class="job-title">{medal} {meta.get('title', 'N/A')}</div>
    <div class="job-company">🏢 {meta.get('company', 'N/A')} &nbsp;|&nbsp; 📍 {meta.get('location', 'N/A')} &nbsp;|&nbsp; 💼 {meta.get('experience', 'N/A')}</div>
    <br/>
    <div>{skill_tags}</div>
    <br/>
    <div style="color: #aaa; font-size: 0.85rem;">{meta.get('description', '')[:180]}…</div>
</div>
""", unsafe_allow_html=True)

    # ── Stats ─────────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📊 Match Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jobs Searched", "100")
    c2.metric("Top Match", f"{int(results[0].get('similarity', 0) * 100)}%")
    c3.metric("Avg Match Score", f"{int(sum(r.get('similarity',0) for r in results)/len(results)*100)}%")
    c4.metric("Search Engine", "Endee Vector DB")

elif not match_btn:
    st.markdown("""
<div style="text-align:center; padding: 3rem; color: #444;">
    <div style="font-size: 3rem;">💼</div>
    <div style="font-size: 1.1rem; margin-top: 1rem;">Upload your resume above and click <b style="color:#f9a825">Find Matching Jobs</b></div>
    <div style="font-size: 0.85rem; margin-top: 0.5rem; font-family: monospace;">Powered by Endee vector similarity search</div>
</div>
""", unsafe_allow_html=True)

st.divider()
st.markdown('<p style="text-align:center; color:#333; font-size:0.8rem; font-family:monospace;">Built with Endee Vector DB · sentence-transformers · Groq LLaMA3 · Streamlit</p>', unsafe_allow_html=True)
