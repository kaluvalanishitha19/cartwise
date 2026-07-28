# cartwise

An AI customer-support agent for e-commerce. RAG over store policies with
citations, order tools (lookup / cancel / refund) with confirmation steps, and
confidence-based escalation to a human agent queue — built with LangGraph,
FastAPI, PostgreSQL + pgvector, and React.

> Status: Week 1 — foundation. Schema, seed data, and policy corpus.

## Quickstart

```bash
docker compose up -d                     # Postgres 16 + pgvector, schema auto-applied
pip install -r backend/requirements.txt
python data/seed.py                      # 30 products, 15 customers, 50 orders w/ event trails
```

Sanity check:

```bash
docker exec -it cartwise-db psql -U cartwise -d cartwise \
  -c "SELECT event_type, detail, occurred_at FROM order_events WHERE order_id = 7 ORDER BY occurred_at;"
```

You should see a believable order timeline, not just a status word.

## Layout

```
backend/   FastAPI app (coming: chat endpoint, LangGraph agent)
data/      schema.sql, seed.py, policies/ (RAG corpus)
frontend/  React + TS chat UI (coming)
```

## TODO (week 1)

- [x] Schema + docker-compose + seed data
- [ ] Policy docs: returns, shipping, refunds (data/policies/)
- [ ] Chunk + embed policies into pgvector
- [ ] FastAPI chat endpoint with streaming
- [ ] React chat UI
- [ ] RAG answers with citations
