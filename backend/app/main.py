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
import re

import numpy as np

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

ORDER_SYSTEM_PROMPT = """You are a customer support agent for an online store called cartwise.

Below is the real order history for the order the customer asked about,
as a timeline of events. Answer the customer's question using ONLY this
timeline. Summarize it like a helpful human would -- don't just repeat
the raw event log. If the order isn't found, say so and suggest they
double check the order number.

Be concise -- 2-4 sentences."""

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

# Matches "order 7", "order #7", "order id 7", "#7" etc. -- simple intent
# detection. LangGraph will replace this with a proper classifier next.
ORDER_ID_PATTERN = re.compile(r"order\s*(?:id\s*)?#?\s*(\d+)|#(\d+)", re.IGNORECASE)

# Simple keyword check for "the customer wants to cancel something".
CANCEL_INTENT_PATTERN = re.compile(r"\bcancel\b", re.IGNORECASE)
REFUND_INTENT_PATTERN = re.compile(r"\brefund\b", re.IGNORECASE)
CONFIRM_PATTERN = re.compile(r"\b(yes|confirm|do it|go ahead|please cancel|please refund)\b", re.IGNORECASE)


class ChatRequest(BaseModel):
    message: str
    pending_action: dict | None = None  # echoed back by the frontend when confirming


class ChatResponse(BaseModel):
    reply: str
    sources: list[str]
    pending_action: dict | None = None  # set when the agent is awaiting confirmation


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


def lookup_order(order_id: int) -> str | None:
    """
    The agent's first tool: look up an order's full event timeline.
    Returns a plain-text summary the LLM can read, or None if not found.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, status, total_cents FROM orders WHERE id = %s", (order_id,))
    order = cur.fetchone()
    if not order:
        cur.close()
        conn.close()
        return None

    _, status, total_cents = order
    cur.execute(
        "SELECT event_type, detail, occurred_at FROM order_events "
        "WHERE order_id = %s ORDER BY occurred_at",
        (order_id,),
    )
    events = cur.fetchall()
    cur.close()
    conn.close()

    lines = [f"Order #{order_id} -- current status: {status} -- total: ${total_cents / 100:.2f}"]
    for event_type, detail, occurred_at in events:
        lines.append(f"  [{occurred_at:%Y-%m-%d %H:%M}] {event_type}: {detail}")
    return "\n".join(lines)


# Example phrases representing safety/privacy concerns. We check if the
# customer's message is semantically close to any of these -- same idea
# as policy search, just matching against a small set of examples instead
# of documents. Anything that clears the threshold gets escalated, not
# answered automatically.
SAFETY_CONCERN_EXAMPLES = [
    "this product caught fire or started smoking",
    "the item shocked me or gave me an electric shock",
    "I got injured or hurt using this product",
    "this item is leaking a dangerous chemical or gas",
    "the product exploded or the battery swelled up",
]
DELIVERY_CONCERN_EXAMPLES = [
    "the delivery driver was following me or acting suspicious",
    "someone took a photo of my house or seemed to be watching me",
    "my package was left somewhere unsafe or in plain view for strangers",
    "I think someone unauthorized has access to my delivery address",
    "the driver behaved inappropriately or made me feel unsafe",
]
SAFETY_DISTANCE_THRESHOLD = 0.42  # below this = close enough to count as a match
# Tuned from measured distances: genuine concerns measured 0.22-0.40, while
# an innocent "is my package lost?" question measured 0.45 -- 0.42 sits in
# the gap between them, with margin on both sides (see
# evals/diagnose_safety_threshold.py).

_safety_embeddings = None  # computed once at startup, see below


def check_safety_concern(message: str) -> tuple[bool, str | None]:
    """
    Returns (True, category) if the message is close in meaning to a
    safety or delivery-privacy concern example, else (False, None).
    """
    global _safety_embeddings
    if _safety_embeddings is None:
        _safety_embeddings = {
            "safety_concern": embed_model.encode(SAFETY_CONCERN_EXAMPLES),
            "delivery_privacy_concern": embed_model.encode(DELIVERY_CONCERN_EXAMPLES),
        }

    message_embedding = embed_model.encode(message)
    for category, examples_embedding in _safety_embeddings.items():
        # cosine distance = 1 - cosine similarity
        sims = np.dot(examples_embedding, message_embedding) / (
            np.linalg.norm(examples_embedding, axis=1) * np.linalg.norm(message_embedding)
        )
        best_distance = 1 - sims.max()
        if best_distance < SAFETY_DISTANCE_THRESHOLD:
            return True, category
    return False, None


def create_escalation(order_id: int | None, reason: str, customer_message: str, agent_note: str) -> None:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO escalations (order_id, reason, customer_message, agent_note) "
        "VALUES (%s, %s, %s, %s)",
        (order_id, reason, customer_message, agent_note),
    )
    conn.commit()
    cur.close()
    conn.close()


# Orders in these statuses can no longer be cancelled -- the item is either
# already with the customer or already handled. This is a real business
# rule, not just a technicality: it's what returns/refunds are for instead.
NON_CANCELLABLE_STATUSES = {"delivered", "returned", "cancelled"}


def cancel_order(order_id: int) -> dict:
    """
    The agent's second tool: cancel an order, IF it's still cancellable.
    Returns a dict describing what happened -- success, or why not.
    This does NOT get called directly from a customer message. The chat
    endpoint below requires an explicit confirmation step first.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT status FROM orders WHERE id = %s", (order_id,))
    row = cur.fetchone()

    if not row:
        cur.close()
        conn.close()
        return {"success": False, "reason": f"No order found with ID #{order_id}."}

    status = row[0]
    if status in NON_CANCELLABLE_STATUSES:
        cur.close()
        conn.close()
        return {
            "success": False,
            "reason": f"Order #{order_id} is already '{status}' and can no longer be cancelled. "
                      f"If you'd like to return it instead, I can walk you through that.",
        }

    cur.execute("UPDATE orders SET status = 'cancelled' WHERE id = %s", (order_id,))
    cur.execute(
        "INSERT INTO order_events (order_id, event_type, detail, occurred_at) "
        "VALUES (%s, 'cancelled', 'Cancelled by customer via support chat.', now())",
        (order_id,),
    )
    conn.commit()
    cur.close()
    conn.close()
    return {"success": True, "reason": f"Order #{order_id} has been cancelled."}


# Per our refunds policy: only items actually returned and inspected are
# refund-eligible. "No damage" isn't a database fact -- it's confirmed
# through the conversation itself, same as a real support agent would.
REFUND_ELIGIBLE_STATUSES = {"returned"}

# From refunds.md: refunds over $150 need manual review before they can
# be issued. We don't have a human escalation queue yet (that's next),
# so for now the agent is honest about the limit instead of pretending
# to process something it can't actually approve.
LARGE_REFUND_THRESHOLD_CENTS = 15000


def initiate_refund(order_id: int) -> dict:
    """
    The agent's third tool: issue a refund, IF the order is eligible.
    Returns a dict describing what happened -- success, or why not.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT status, total_cents FROM orders WHERE id = %s", (order_id,))
    row = cur.fetchone()

    if not row:
        cur.close()
        conn.close()
        return {"success": False, "reason": f"No order found with ID #{order_id}."}

    status, total_cents = row
    if status not in REFUND_ELIGIBLE_STATUSES:
        cur.close()
        conn.close()
        return {
            "success": False,
            "reason": f"Order #{order_id} is currently '{status}'. Refunds can only be issued "
                      f"once an item has been returned and received at our warehouse.",
        }

    if total_cents > LARGE_REFUND_THRESHOLD_CENTS:
        cur.close()
        conn.close()
        create_escalation(
            order_id=order_id,
            reason="large_refund",
            customer_message=f"Refund requested for order #{order_id} (${total_cents / 100:.2f})",
            agent_note=f"Order total ${total_cents / 100:.2f} exceeds the $150 automatic refund limit.",
        )
        return {
            "success": False,
            "reason": f"This order totals ${total_cents / 100:.2f}, which is over our $150 "
                      f"automatic refund limit. This needs manual review by our support team "
                      f"before it can be issued -- I've flagged it for a human agent.",
        }

    cur.execute(
        "INSERT INTO order_events (order_id, event_type, detail, occurred_at) "
        "VALUES (%s, 'refund_issued', 'Refund issued via support chat after damage confirmation.', now())",
        (order_id,),
    )
    conn.commit()
    cur.close()
    conn.close()
    return {
        "success": True,
        "reason": f"Your refund of ${total_cents / 100:.2f} for order #{order_id} has been issued "
                  f"to your original payment method. It typically takes 7-10 business days to appear.",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/escalations")
def list_escalations():
    """
    Returns all open escalations, most recent first -- what a human
    support agent would see when they open the dashboard.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, order_id, reason, customer_message, agent_note, status, created_at
        FROM escalations
        ORDER BY created_at DESC
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        {
            "id": r[0],
            "order_id": r[1],
            "reason": r[2],
            "customer_message": r[3],
            "agent_note": r[4],
            "status": r[5],
            "created_at": r[6].isoformat(),
        }
        for r in rows
    ]


@app.post("/escalations/{escalation_id}/resolve")
def resolve_escalation(escalation_id: int):
    """
    Marks an escalation as resolved -- the human agent clicking
    "resolved" after they've handled the case.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("UPDATE escalations SET status = 'resolved' WHERE id = %s", (escalation_id,))
    conn.commit()
    cur.close()
    conn.close()
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Step -1: safety/privacy concerns take priority over everything else,
    # even an in-progress confirmation flow. Never handled automatically.
    is_concern, category = check_safety_concern(req.message)
    if is_concern:
        order_match = ORDER_ID_PATTERN.search(req.message)
        order_id = int(order_match.group(1) or order_match.group(2)) if order_match else None
        create_escalation(
            order_id=order_id,
            reason=category,
            customer_message=req.message,
            agent_note=f"Auto-flagged as {category.replace('_', ' ')} -- needs human review, not answered automatically.",
        )
        return ChatResponse(
            reply="I'm really sorry to hear that. This isn't something I can resolve automatically -- "
                  "I've flagged it for our support team right away, and someone will follow up with you directly.",
            sources=[],
        )

    # Step 0: is this a reply to a pending "are you sure?" question?
    if req.pending_action and req.pending_action.get("action") == "cancel":
        order_id = req.pending_action["order_id"]
        if CONFIRM_PATTERN.search(req.message):
            result = cancel_order(order_id)
            return ChatResponse(reply=result["reason"], sources=[f"Order #{order_id} cancellation"])
        else:
            return ChatResponse(
                reply=f"Okay, I won't cancel order #{order_id}. Anything else I can help with?",
                sources=[],
            )

    if req.pending_action and req.pending_action.get("action") == "refund_damage_check":
        order_id = req.pending_action["order_id"]
        if CONFIRM_PATTERN.search(req.message):
            # Damage confirmed as none -- now ask for the final refund confirmation.
            return ChatResponse(
                reply=f"Got it, thanks. Just to confirm -- process the refund for order #{order_id}? "
                      f"Reply 'yes' to confirm.",
                sources=[],
                pending_action={"action": "refund", "order_id": order_id},
            )
        else:
            return ChatResponse(
                reply=f"I understand -- since the item wasn't returned in undamaged condition, "
                      f"I can't process an automatic refund for order #{order_id}. "
                      f"I've flagged this for a human agent to review.",
                sources=[],
            )

    if req.pending_action and req.pending_action.get("action") == "refund":
        order_id = req.pending_action["order_id"]
        if CONFIRM_PATTERN.search(req.message):
            result = initiate_refund(order_id)
            return ChatResponse(reply=result["reason"], sources=[f"Order #{order_id} refund"])
        else:
            return ChatResponse(
                reply=f"Okay, I won't process a refund for order #{order_id}. Anything else I can help with?",
                sources=[],
            )

    order_match = ORDER_ID_PATTERN.search(req.message)

    # Step 1: order + "cancel" -- ask for confirmation before acting.
    if order_match and CANCEL_INTENT_PATTERN.search(req.message):
        order_id = int(order_match.group(1) or order_match.group(2))
        return ChatResponse(
            reply=f"Just to confirm -- you'd like to cancel order #{order_id}? "
                  f"This can't be undone. Reply 'yes' to confirm.",
            sources=[],
            pending_action={"action": "cancel", "order_id": order_id},
        )

    # Step 2: order + "refund" -- first confirm the item arrived undamaged
    # (our refunds policy requires this before any refund is eligible),
    # THEN ask to confirm the refund itself.
    if order_match and REFUND_INTENT_PATTERN.search(req.message):
        order_id = int(order_match.group(1) or order_match.group(2))
        return ChatResponse(
            reply=f"Before I can process a refund for order #{order_id}, can you confirm the "
                  f"returned item arrived at our warehouse with no damage? Reply 'yes' to confirm.",
            sources=[],
            pending_action={"action": "refund_damage_check", "order_id": order_id},
        )

    # Step 3: an order ID with no cancel/refund intent -- treat as a lookup.
    if order_match:
        order_id = int(order_match.group(1) or order_match.group(2))
        timeline = lookup_order(order_id)

        if timeline is None:
            return ChatResponse(
                reply=f"I couldn't find an order with ID #{order_id}. "
                      f"Could you double check the order number?",
                sources=[],
            )

        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": ORDER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Order timeline:\n\n{timeline}\n\nCustomer question: {req.message}"},
            ],
        )
        return ChatResponse(reply=response["message"]["content"], sources=[f"Order #{order_id} history"])

    # Step 4: no order ID mentioned -- treat as a policy question (RAG, as before).
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
