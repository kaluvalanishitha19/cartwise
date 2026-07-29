"""
cartwise backend API.

Wraps the same search + generate logic from data/chat_test.py behind a
real HTTP endpoint, so a website (or anything else) can talk to it.

Run with:
    uvicorn app.main:app --reload --port 8000

Then test with:
    curl -X POST http://localhost:8000/chat \
      -H "Content-Type: application/json" \
      -d '{"message": "can I return a clearance item after 10 days?"}'
"""

import os

import ollama
import psycopg2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://cartwise:cartwise_dev@localhost:5433/cartwise"
)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
TOP_K = 3

SYSTEM_PROMPT = """You are a customer support agent for an online store called cartwise.

Answer the customer's question using ONLY the policy excerpts provided below.
Do not use any outside knowledge. If the excerpts don't contain the answer,
say you don't have that information and the customer should contact support.

At the end of your answer, on a new line, cite which policy section(s) you
used, like: [Source: Returns - Clearance items]

Be concise -- 2-4 sentences before the citation."""

app = FastAPI(title="cartwise API")

# Lets your React frontend (running on a different port) call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local dev; we'll lock this down before deploying
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the embedding model once at startup, not on every request -- loading
# it per-request would make every chat message slow.
embed_model = SentenceTransformer(EMBED_MODEL_NAME)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    sources: list[str]


def get_db_connection():
    return psycopg2.connect(DATABASE_URL)


def retrieve_chunks(question: str) -> list[tuple]:
    conn = get_db_connection()
    cur = conn.cursor()
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
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results


def build_context(chunks: list[tuple]) -> tuple[str, list[str]]:
    context_parts = []
    sources = []
    for doc_title, idx, text, _ in chunks:
        # First line of a chunk is its "## Section Name" header -- use it
        # as a readable source label instead of a bare index number.
        section_name = text.split("\n")[0].lstrip("# ").strip()
        label = f"{doc_title} - {section_name}"
        sources.append(label)
        context_parts.append(f"[{label}]\n{text}")
    return "\n\n".join(context_parts), sources


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    chunks = retrieve_chunks(req.message)
    context, sources = build_context(chunks)

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Policy excerpts:\n\n{context}\n\nCustomer question: {req.message}"},
        ],
    )

    return ChatResponse(reply=response["message"]["content"], sources=sources)