# Cartwise

An AI customer-support agent for e-commerce built end to end, running entirely
free and local. It answers policy questions with citations, looks up and acts on
real orders, and knows when to hand off to a human instead of guessing.

**Stack:** React + TypeScript · FastAPI · PostgreSQL + pgvector · sentence-transformers
(local embeddings) · Ollama (local LLM) no paid API keys required to run this project.

---

## What it does

**Answers policy questions with citations (RAG).** Store policies (returns,
shipping, refunds) are chunked by topic, embedded, and stored in pgvector. A
customer question retrieves the most relevant policy section and the agent
answers using *only* that text — every answer names its source.

**Looks up real orders.** Ask about an order and the agent pulls its full event
timeline (placed → shipped → delayed → delivered, etc.) from Postgres and
summarizes it in plain language.

**Cancels and refunds orders safely.** Both actions require an explicit
confirmation step before anything happens. Cancellation is blocked once an
order is delivered or returned. Refunds are only eligible for orders already
returned, require a damage confirmation first, and anything over $150 is
automatically flagged for manual review rather than auto-approved.

**Escalates instead of guessing.** Safety concerns ("this product caught
fire") and delivery privacy concerns ("the driver was watching my house") are
detected via semantic similarity to example phrases and immediately routed to
a human queue — never answered automatically. A small agent dashboard lets a
human view and resolve open escalations.

**Is tested, not just demoed.** A 10-case eval suite (`evals/run_evals.py`)
checks retrieval correctness, tool behavior, and escalation triggers against
the live API — not exact wording, but the actual decisions made. The safety
escalation threshold was tuned using measured embedding distances after the
eval suite caught a real false-positive bug (see `evals/diagnose_safety_threshold.py`).

## Architecture

```
React chat UI
      │
FastAPI /chat endpoint  ──▶  routes each message:
      │
      ├── Safety/privacy concern? ──▶ escalate immediately (skips everything else)
      ├── "cancel order N"        ──▶ confirm, then cancel_order()
      ├── "refund order N"        ──▶ confirm damage, confirm refund, then initiate_refund()
      ├── "order N" (lookup)      ──▶ lookup_order() → real event timeline
      └── anything else           ──▶ pgvector search → policy chunk(s)
                                          │
                                          ▼
                                  Ollama (local LLM)
                                          │
                                          ▼
                                Answer + citation
```

## Why local, not OpenAI

This project runs entirely on free, local infrastructure by design:
`sentence-transformers` for embeddings (384-dim, `all-MiniLM-L6-v2`) and
`Ollama` running `llama3.2` for generation — no API key, no per-request cost,
nothing leaves the machine. It's also a legitimate production pattern
(cost control, data privacy), not just a budget workaround.

## Running it locally

```bash
# 1. Database
docker compose up -d
python3 data/seed.py                  # seed products, customers, orders
python3 data/ingest.py                # chunk + embed policy docs

# 2. Backend (requires Ollama running with `ollama pull llama3.2`)
cd backend
pip3 install -r requirements.txt
DATABASE_URL="postgresql://cartwise:cartwise_dev@localhost:5433/cartwise" \
  uvicorn app.main:app --reload --port 8000

# 3. Frontend
cd frontend
npm install
npm run dev   # http://localhost:5173

# 4. Evals (backend must be running)
python3 evals/run_evals.py
```

## Project layout

```
backend/   FastAPI app — RAG, order tools, escalation logic
data/      schema.sql, seed.py, ingest.py, policies/ (the RAG corpus)
frontend/  React + TypeScript chat UI + agent dashboard
evals/     automated eval suite + threshold diagnostics
```

