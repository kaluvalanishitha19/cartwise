"""
cartwise search test.

Takes a question, embeds it with the same free model used for ingestion,
and finds the closest policy chunks using pgvector's similarity search.

This is a standalone test script -- the same logic becomes part of the
/chat endpoint next.

Usage:
    python3 data/search_test.py "can I return a clearance item after 10 days?"
"""

import os
import sys
import psycopg2
from sentence_transformers import SentenceTransformer

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://cartwise:cartwise_dev@localhost:5433/cartwise"
)
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3  # how many chunks to retrieve


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 data/search_test.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]
    print(f"Question: {question}\n")

    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode(question).tolist()

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # <=> is pgvector's cosine distance operator: smaller = more similar.
    cur.execute(
        """
        SELECT doc_title, chunk_index, chunk_text,
               embedding <=> %s::vector AS distance
        FROM policy_chunks
        ORDER BY distance ASC
        LIMIT %s;
        """,
        (query_embedding, TOP_K),
    )
    results = cur.fetchall()

    print(f"Top {TOP_K} matching chunks:\n")
    for doc_title, chunk_index, chunk_text, distance in results:
        print(f"--- {doc_title} [{chunk_index}]  (distance: {distance:.4f}) ---")
        print(chunk_text)
        print()

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()