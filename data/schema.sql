-- cartwise schema
-- Runs automatically on first `docker compose up` via docker-entrypoint-initdb.d

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE customers (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  email TEXT UNIQUE NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE products (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  category TEXT NOT NULL,
  price_cents INT NOT NULL CHECK (price_cents >= 0),
  description TEXT,
  return_window_days INT NOT NULL DEFAULT 30
);

CREATE TABLE orders (
  id SERIAL PRIMARY KEY,
  customer_id INT NOT NULL REFERENCES customers(id),
  -- placed | shipped | delayed | delivered | cancelled | returned
  status TEXT NOT NULL CHECK (status IN
    ('placed','shipped','delayed','delivered','cancelled','returned')),
  total_cents INT NOT NULL CHECK (total_cents >= 0),
  placed_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE order_items (
  id SERIAL PRIMARY KEY,
  order_id INT NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
  product_id INT NOT NULL REFERENCES products(id),
  quantity INT NOT NULL CHECK (quantity > 0),
  unit_price_cents INT NOT NULL CHECK (unit_price_cents >= 0)
);

-- Why a separate events table? (You owe me this answer out loud.)
CREATE TABLE order_events (
  id SERIAL PRIMARY KEY,
  order_id INT NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
  -- placed, payment_confirmed, shipped, delay_reported, delivered,
  -- cancelled, return_requested, returned, refund_issued
  event_type TEXT NOT NULL,
  detail TEXT,
  occurred_at TIMESTAMPTZ NOT NULL
);

-- RAG store for policy documents.
-- 1536 dims = OpenAI text-embedding-3-small. Embeddings are filled in
-- by the ingestion script (day 2); seed.py leaves this table empty.
CREATE TABLE policy_chunks (
  id SERIAL PRIMARY KEY,
  doc_title TEXT NOT NULL,
  chunk_text TEXT NOT NULL,
  chunk_index INT NOT NULL,
  embedding vector(1536)
);

CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_order_items_order ON order_items(order_id);
CREATE INDEX idx_order_events_order ON order_events(order_id, occurred_at);
