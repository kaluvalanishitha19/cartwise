"""
cartwise seed script.

Creates a believable mock storefront:
  - handcrafted product catalog (~30 products, 5 categories)
  - 15 customers (Faker)
  - 50 orders whose statuses each have a matching, time-ordered event trail

Usage:
    docker compose up -d
    pip install -r backend/requirements.txt
    python data/seed.py

Idempotent-ish: truncates the transactional tables first, so it's safe to re-run.
"""

import os
import random
from datetime import datetime, timedelta, timezone

import psycopg2
from faker import Faker

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://cartwise:cartwise_dev@localhost:5432/cartwise"
)

random.seed(42)
fake = Faker()
Faker.seed(42)

# ---------------------------------------------------------------------------
# Handcrafted catalog. Real-sounding names make every demo screenshot better.
# (name, category, price_cents, description, return_window_days)
# Note the deliberate policy variety: electronics 30d, clearance 7d, etc.
# ---------------------------------------------------------------------------
PRODUCTS = [
    # Electronics (30-day returns)
    ("Aurora ANC Wireless Headphones", "electronics", 12999, "Over-ear, active noise cancelling, 40h battery.", 30),
    ("PulseBeat True Wireless Earbuds", "electronics", 5999, "Compact earbuds with wireless charging case.", 30),
    ("VoltEdge 65W GaN Charger", "electronics", 3499, "Dual USB-C fast charger for laptops and phones.", 30),
    ("ClearCast 1080p Webcam", "electronics", 4499, "Auto-focus webcam with built-in privacy shutter.", 30),
    ("NimbusPad Wireless Charging Stand", "electronics", 2799, "15W Qi stand, phone and earbuds together.", 30),
    ("EchoBar Mini Bluetooth Speaker", "electronics", 3999, "Room-filling sound, IPX5 splash resistant.", 30),
    ("TrailCam 4K Action Camera", "electronics", 15999, "Waterproof to 10m, image-stabilized 4K60.", 30),
    # Home & Kitchen (30-day returns)
    ("BrewCraft 12-Cup Drip Coffee Maker", "home_kitchen", 6499, "Programmable, thermal carafe, bold setting.", 30),
    ("SearMaster Cast Iron Skillet 10\"", "home_kitchen", 2999, "Pre-seasoned cast iron, oven safe to 500F.", 30),
    ("AeroChop Immersion Blender", "home_kitchen", 4299, "3-speed hand blender with whisk attachment.", 30),
    ("CrispAir 5.5Qt Air Fryer", "home_kitchen", 8999, "Digital presets, dishwasher-safe basket.", 30),
    ("PureDrop Water Filter Pitcher", "home_kitchen", 3299, "10-cup pitcher, 3-month filter included.", 30),
    ("CozyGlow Ceramic Space Heater", "home_kitchen", 4999, "1500W with tip-over auto shutoff.", 30),
    # Fitness (30-day returns)
    ("FlexForm Yoga Mat 6mm", "fitness", 2599, "Non-slip TPE mat with carry strap.", 30),
    ("IronGrip Adjustable Dumbbell 25lb", "fitness", 10999, "Dial-adjust 5-25lb, single dumbbell.", 30),
    ("StrideTrack Fitness Band", "fitness", 7999, "Heart rate, sleep tracking, 7-day battery.", 30),
    ("CorePro Resistance Band Set", "fitness", 1999, "5 bands, door anchor, handles, carry bag.", 30),
    ("HydraFuel 32oz Insulated Bottle", "fitness", 2299, "Keeps drinks cold 24h, leakproof lid.", 30),
    ("BalanceWorks Foam Roller", "fitness", 1899, "High-density roller for recovery.", 30),
    # Accessories (30-day returns)
    ("UrbanShield Laptop Backpack", "accessories", 5499, "Fits 15.6\" laptops, water resistant, USB port.", 30),
    ("SlimVault RFID Wallet", "accessories", 2499, "Minimalist aluminum card holder.", 30),
    ("SunTrek Polarized Sunglasses", "accessories", 3499, "UV400 polarized lenses, flexible frame.", 30),
    ("NomadStrap Quick-Release Watch Band", "accessories", 1599, "Silicone band, 20mm/22mm sizes.", 30),
    ("PackRight Packing Cubes (6pc)", "accessories", 2899, "Compression cubes for carry-on travel.", 30),
    # Clearance (7-day returns -- deliberate policy conflict for RAG demos)
    ("Clearance: RetroWave Desk Lamp", "clearance", 1499, "Final-sale styling lamp, warm LED.", 7),
    ("Clearance: SnapFit Phone Case (Model X9)", "clearance", 799, "Discontinued model case, assorted colors.", 7),
    ("Clearance: WinterPeak Beanie", "clearance", 899, "Last season's colorway.", 7),
    ("Clearance: DeskMate Monitor Riser", "clearance", 1999, "Bamboo riser, minor cosmetic blemishes.", 7),
    ("Clearance: GlowRing Selfie Light", "clearance", 1299, "Clip-on ring light, USB powered.", 7),
    ("Clearance: TidyDesk Cable Organizer Kit", "clearance", 999, "Assorted clips and sleeves.", 7),
]

# status -> ordered event trail (event_type, detail template)
EVENT_TRAILS = {
    "placed": [
        ("placed", "Order received."),
        ("payment_confirmed", "Payment confirmed via card ending {last4}."),
    ],
    "shipped": [
        ("placed", "Order received."),
        ("payment_confirmed", "Payment confirmed via card ending {last4}."),
        ("shipped", "Shipped via {carrier}, tracking {tracking}."),
    ],
    "delayed": [
        ("placed", "Order received."),
        ("payment_confirmed", "Payment confirmed via card ending {last4}."),
        ("shipped", "Shipped via {carrier}, tracking {tracking}."),
        ("delay_reported", "Carrier reported a delay: {delay_reason}"),
    ],
    "delivered": [
        ("placed", "Order received."),
        ("payment_confirmed", "Payment confirmed via card ending {last4}."),
        ("shipped", "Shipped via {carrier}, tracking {tracking}."),
        ("delivered", "Delivered - left at front door."),
    ],
    "cancelled": [
        ("placed", "Order received."),
        ("payment_confirmed", "Payment confirmed via card ending {last4}."),
        ("cancelled", "Cancelled by customer before shipment. Refund issued to original payment method."),
    ],
    "returned": [
        ("placed", "Order received."),
        ("payment_confirmed", "Payment confirmed via card ending {last4}."),
        ("shipped", "Shipped via {carrier}, tracking {tracking}."),
        ("delivered", "Delivered - left at front door."),
        ("return_requested", "Customer requested a return: {return_reason}"),
        ("returned", "Return received at warehouse. Refund of {refund} issued."),
    ],
}

# Weighted so the interesting support cases (delayed, returned) are well represented.
STATUS_WEIGHTS = {
    "placed": 8, "shipped": 10, "delayed": 8,
    "delivered": 14, "cancelled": 4, "returned": 6,
}

CARRIERS = ["UPS", "FedEx", "USPS"]
DELAY_REASONS = [
    "weather conditions at regional hub.",
    "package missorted at distribution center.",
    "unexpected carrier volume; new ETA pending.",
]
RETURN_REASONS = [
    "item arrived damaged.",
    "wrong size/fit.",
    "changed mind.",
    "item not as described.",
]


def main() -> None:
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur = conn.cursor()

    cur.execute(
        "TRUNCATE order_events, order_items, orders, customers, products "
        "RESTART IDENTITY CASCADE;"
    )

    # products
    for name, cat, price, desc, window in PRODUCTS:
        cur.execute(
            "INSERT INTO products (name, category, price_cents, description, return_window_days) "
            "VALUES (%s,%s,%s,%s,%s)",
            (name, cat, price, desc, window),
        )

    # customers
    customer_ids = []
    for _ in range(15):
        cur.execute(
            "INSERT INTO customers (name, email) VALUES (%s,%s) RETURNING id",
            (fake.name(), fake.unique.email()),
        )
        customer_ids.append(cur.fetchone()[0])

    statuses = [s for s, w in STATUS_WEIGHTS.items() for _ in range(w)]
    now = datetime.now(timezone.utc)

    for _ in range(50):
        status = random.choice(statuses)
        customer_id = random.choice(customer_ids)
        placed_at = now - timedelta(
            days=random.randint(2, 90),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )

        # 1-3 line items
        items = random.sample(range(1, len(PRODUCTS) + 1), k=random.randint(1, 3))
        total = 0
        line_rows = []
        for pid in items:
            qty = random.randint(1, 2)
            price = PRODUCTS[pid - 1][2]
            total += qty * price
            line_rows.append((pid, qty, price))

        cur.execute(
            "INSERT INTO orders (customer_id, status, total_cents, placed_at) "
            "VALUES (%s,%s,%s,%s) RETURNING id",
            (customer_id, status, total, placed_at),
        )
        order_id = cur.fetchone()[0]

        for pid, qty, price in line_rows:
            cur.execute(
                "INSERT INTO order_items (order_id, product_id, quantity, unit_price_cents) "
                "VALUES (%s,%s,%s,%s)",
                (order_id, pid, qty, price),
            )

        # event trail: sequential timestamps, hours-to-days apart
        ctx = {
            "last4": f"{random.randint(1000, 9999)}",
            "carrier": random.choice(CARRIERS),
            "tracking": f"1Z{random.randint(10**8, 10**9 - 1)}",
            "delay_reason": random.choice(DELAY_REASONS),
            "return_reason": random.choice(RETURN_REASONS),
            "refund": f"${total / 100:.2f}",
        }
        ts = placed_at
        for event_type, detail_tpl in EVENT_TRAILS[status]:
            cur.execute(
                "INSERT INTO order_events (order_id, event_type, detail, occurred_at) "
                "VALUES (%s,%s,%s,%s)",
                (order_id, event_type, detail_tpl.format(**ctx), ts),
            )
            ts += timedelta(hours=random.randint(4, 60))

    conn.commit()

    cur.execute("SELECT count(*) FROM orders")
    n_orders = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM order_events")
    n_events = cur.fetchone()[0]
    print(f"Seeded {len(PRODUCTS)} products, {len(customer_ids)} customers, "
          f"{n_orders} orders, {n_events} order events.")
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
