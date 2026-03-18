"""
ingest_jobs.py
--------------
Indexes 100 job descriptions into Endee vector DB.

Usage:
    python ingest_jobs.py
"""

from sentence_transformers import SentenceTransformer # type: ignore
from endee import Endee, Precision # type: ignore
from jobs_data import JOBS

INDEX_NAME    = "job_listings"
EMBEDDING_DIM = 384
ENDEE_HOST    = "https://endee-server-production-625c.up.railway.app"


def job_to_text(job: dict) -> str:
    """Convert a job dict into a rich text string for embedding."""
    skills = ", ".join(job["skills"])
    return (
        f"Title: {job['title']}\n"
        f"Company: {job['company']}\n"
        f"Location: {job['location']}\n"
        f"Experience: {job['experience']}\n"
        f"Skills: {skills}\n"
        f"Description: {job['description']}"
    )


def main():
    print("🔌 Connecting to Endee vector DB …")
    client = Endee()
    client.set_base_url(f"{ENDEE_HOST}/api/v1")

    # Delete existing index if present
    try:
        client.delete_index(INDEX_NAME)
        print(f"   Deleted existing index '{INDEX_NAME}'")
    except Exception:
        pass

    client.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        space_type="cosine",
        precision=Precision.INT8,
    )
    print(f"   ✅ Index '{INDEX_NAME}' created (dim={EMBEDDING_DIM}, cosine, INT8)")

    index = client.get_index(name=INDEX_NAME)

    print("🤖 Loading embedding model …")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"📋 Indexing {len(JOBS)} job listings …")
    texts  = [job_to_text(j) for j in JOBS]
    vectors = model.encode(texts, show_progress_bar=True).tolist()

    items = []
    for job, vec in zip(JOBS, vectors):
        items.append({
            "id": job["id"],
            "vector": vec,
            "meta": {
                "title":      job["title"],
                "company":    job["company"],
                "location":   job["location"],
                "experience": job["experience"],
                "skills":     ", ".join(job["skills"]),
                "description": job["description"],
            },
        })

    index.upsert(items)
    print(f"\n✅ Done! Indexed {len(JOBS)} jobs into Endee.")


if __name__ == "__main__":
    main()
