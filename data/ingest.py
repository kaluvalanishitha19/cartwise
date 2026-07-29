"""
cartwise ingestion script.

Reads the policy docs, splits each one into chunks by markdown ## header
(one chunk = one complete rule/topic), embeds each chunk with a free local
model, and stores them in policy_chunks.

Usage:
    python3 data/ingest.py
"""

import os
import re
import psycopg2
from sentence_transformers import SentenceTransformer

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://cartwise:cartwise_dev@localhost:5433/cartwise"
)
POLICY_DIR = os.path.join(os.path.dirname(__file__), "policies")

# Free, local, 384-dimension embedding model. Downloads once (~80MB), then cached.
MODEL_NAME = "all-MiniLM-L6-v2"


def split_into_chunks(text: str, doc_title: str) -> list[str]:
    """
    Split a markdown policy doc into chunks by ## header.
    Each chunk = the header line + everything under it, until the next ##.
    This is the "one chunk = one complete rule" idea from our chunking discussion.
    """
    # Split on lines starting with "## " (keep the header attached to its content)
    parts = re.split(r"\n(?=## )", text.strip())
    chunks = []
    for part in parts:
        part = part.strip()
        if not part or part.startswith("# "):  # skip the lone top-level "# Title" line
            continue
        chunks.append(part)
    return chunks


def main() -> None:
    print(f"Loading embedding model ({MODEL_NAME})... first run downloads it, be patient.")
    model = SentenceTransformer(MODEL_NAME)

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # Wipe any previous ingestion so re-running this script is safe.
    cur.execute("TRUNCATE policy_chunks RESTART IDENTITY;")

    total_chunks = 0
    for filename in sorted(os.listdir(POLICY_DIR)):
        if not filename.endswith(".md") or filename == "README.md":
            continue

        doc_title = filename.replace(".md", "").replace("_", " ").title()
        path = os.path.join(POLICY_DIR, filename)
        with open(path, "r") as f:
            text = f.read()

        chunks = split_into_chunks(text, doc_title)
        print(f"\n{filename}: {len(chunks)} chunks")

        for i, chunk_text in enumerate(chunks):
            embedding = model.encode(chunk_text).tolist()
            cur.execute(
                "INSERT INTO policy_chunks (doc_title, chunk_text, chunk_index, embedding) "
                "VALUES (%s, %s, %s, %s)",
                (doc_title, chunk_text, i, embedding),
            )
            first_line = chunk_text.split("\n")[0]
            print(f"  [{i}] {first_line}")
            total_chunks += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"\nDone. {total_chunks} chunks embedded and stored in policy_chunks.")


if __name__ == "__main__":
    main()
