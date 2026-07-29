"""
cartwise RAG chat test.

The full loop:
  1. Take a customer question
  2. Search policy_chunks for the closest matches (what search_test.py did)
  3. Hand those chunks to a local LLM (Ollama) and ask it to answer
     USING ONLY that text, and to say which policy it used

This is the first true end-to-end RAG answer -- the next step after this
is wrapping this same logic in a FastAPI endpoint + chat UI.

Usage:
    python3 data/chat_test.py "can I return a clearance item after 10 days?"
"""

import os
import sys
import psycopg2
import ollama
from sentence_transformers import SentenceTransformer

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://cartwise:cartwise_dev@localhost:5433/cartwise"
)
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
TOP_K = 3

SYSTEM_PROMPT = """You are a customer support agent for an online store called cartwise.

Answer the customer's question using ONLY the policy excerpts provided below.
Do not use any outside knowledge. If the excerpts don't contain the answer,
say you don't have that information and the customer should contact support.

At the end of your answer, on a new line, cite which policy section(s) you
used, like: [Source: Returns - Clearance items]

Be concise -- 2-4 sentences before the citation."""


def retrieve_chunks(question: str, embed_model: SentenceTransformer, cur) -> list[tuple]:
    query_embedding = embed_model.encode(question).tolist()
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
    return cur.fetchall()


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 data/chat_test.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]
    print(f"Customer question: {question}\n")

    print("Searching policies...")
    embed_model = SentenceTransformer(EMBED_MODEL)
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    chunks = retrieve_chunks(question, embed_model, cur)
    cur.close()
    conn.close()

    context = "\n\n".join(
        f"[{doc_title} - section {idx}]\n{text}" for doc_title, idx, text, _ in chunks
    )
    print("Retrieved context:")
    for doc_title, idx, text, distance in chunks:
        first_line = text.split(chr(10))[0]
        print(f"  - {doc_title} [{idx}] {first_line}  (distance: {distance:.4f})")

    print("\nAsking the local model to answer...\n")
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Policy excerpts:\n\n{context}\n\nCustomer question: {question}"},
        ],
    )

    print("=" * 60)
    print("CARTWISE AGENT RESPONSE:")
    print("=" * 60)
    print(response["message"]["content"])


if __name__ == "__main__":
    main()