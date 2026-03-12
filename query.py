"""
query.py
--------
Command-line RAG Q&A interface.
Retrieves relevant chunks from Endee, then asks Gemini to answer.

Usage:
    python query.py
    python query.py --question "What is the refund policy?"

Requirements:
    Set GEMINI_API_KEY environment variable (free at aistudio.google.com)
"""

import os
import argparse
from sentence_transformers import SentenceTransformer
from endee import Endee
import google.generativeai as genai

# ── Config ────────────────────────────────────────────────────────────────────
INDEX_NAME   = "rag_documents"
TOP_K        = 5
ENDEE_HOST   = "http://localhost:8080"
GEMINI_MODEL = "gemini-1.5-flash"
# ──────────────────────────────────────────────────────────────────────────────


def build_clients():
    """Initialise Endee + Gemini clients."""
    # Endee
    endee = Endee()
    endee.set_base_url(f"{ENDEE_HOST}/api/v1")
    index = endee.get_index(name=INDEX_NAME)

    # Gemini
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com"
        )
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel(GEMINI_MODEL)

    # Embedding model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    return index, llm, embedder


def retrieve(index, embedder, question: str, top_k: int = TOP_K):
    """Embed the question and search Endee for similar chunks."""
    q_vec = embedder.encode(question, convert_to_list=True)
    results = index.query(vector=q_vec, top_k=top_k)
    return results


def generate_answer(llm, question: str, context_chunks: list) -> str:
    """Build a prompt with retrieved context and call Gemini."""
    context = "\n\n---\n\n".join(
        f"[Source: {r.meta.get('source', 'unknown')}]\n{r.meta.get('text', '')}"
        for r in context_chunks
    )

    prompt = f"""You are a helpful assistant. Answer the question below using ONLY the provided context.
If the answer is not in the context, say "I couldn't find relevant information in the documents."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    response = llm.generate_content(prompt)
    return response.text.strip()


def answer_question(question: str) -> dict:
    """Full RAG pipeline: retrieve → generate → return."""
    index, llm, embedder = build_clients()

    print(f"\n🔍 Searching for: {question}")
    results = retrieve(index, embedder, question)

    if not results:
        return {"question": question, "answer": "No relevant documents found.", "sources": []}

    sources = list({r.meta.get("source", "unknown") for r in results})
    print(f"   Found {len(results)} relevant chunks from: {sources}")

    print("💬 Generating answer …")
    answer = generate_answer(llm, question, results)

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "chunks": [r.meta.get("text", "") for r in results],
    }


def main():
    parser = argparse.ArgumentParser(description="RAG Q&A with Endee + Gemini")
    parser.add_argument("--question", "-q", type=str, default="", help="Question to ask")
    args = parser.parse_args()

    question = args.question or input("\n❓ Enter your question: ").strip()
    if not question:
        print("No question provided.")
        return

    result = answer_question(question)

    print("\n" + "=" * 60)
    print(f"📖 ANSWER:\n{result['answer']}")
    print(f"\n📎 Sources: {', '.join(result['sources'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
