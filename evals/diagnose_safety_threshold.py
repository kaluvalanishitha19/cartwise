"""
Diagnostic: why did "my package hasn't arrived, is it lost?" get
flagged as a delivery privacy concern? This prints the actual distance
scores so we can pick the right threshold instead of guessing.

Usage:
    python3 evals/diagnose_safety_threshold.py
"""

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

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

TEST_MESSAGES = [
    ("false positive (should NOT escalate)", "my package hasn't arrived in 3 weeks, is it lost?"),
    ("false positive (should NOT escalate)", "when will my order be delivered?"),
    ("genuine positive (SHOULD escalate)", "someone took a photo of my house and it made me feel unsafe"),
    ("genuine positive (SHOULD escalate)", "this product caught fire when I plugged it in"),
]

for label, message in TEST_MESSAGES:
    q_emb = model.encode(message)
    best_overall = 1.0
    best_match = ""
    for category, examples in [("safety", SAFETY_CONCERN_EXAMPLES), ("delivery", DELIVERY_CONCERN_EXAMPLES)]:
        ex_emb = model.encode(examples)
        sims = np.dot(ex_emb, q_emb) / (np.linalg.norm(ex_emb, axis=1) * np.linalg.norm(q_emb))
        distance = 1 - sims.max()
        if distance < best_overall:
            best_overall = distance
            best_match = f"{category}: '{examples[sims.argmax()]}'"

    print(f"[{label}]")
    print(f"  message: \"{message}\"")
    print(f"  closest distance: {best_overall:.3f}  (closest example -> {best_match})")
    print()