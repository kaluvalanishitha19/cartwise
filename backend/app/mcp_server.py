"""
Cartwise MCP server.

Exposes the same order tools already used by the /chat endpoint
(lookup_order, cancel_order, initiate_refund) as MCP tools, so any
MCP-compatible client -- Claude Desktop, a custom agent, the MCP
Inspector -- can call them directly and let an LLM decide when to
use them, instead of the regex-based routing in app/main.py.

This file does NOT duplicate the business logic. It imports the real
functions from app.main and wraps them. If the underlying rules change
(e.g. the $150 refund threshold), this server picks up the change
automatically.

Run with:
    mcp dev backend/app/mcp_server.py

Or point Claude Desktop's config at it (see README section below).
"""

import sys
from pathlib import Path

# 'mcp dev' loads this file directly and only puts its own folder (app/)
# on sys.path -- not backend/, which is what "app.main" needs to resolve
# as a package. Adding backend/ explicitly makes this work the same way
# whether it's run via 'mcp dev', 'python app/mcp_server.py', or imported
# normally as part of the app package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.mcpserver import MCPServer

from app.main import (
    cancel_order as _cancel_order,
    initiate_refund as _initiate_refund,
    lookup_order as _lookup_order,
)

mcp = MCPServer("cartwise")


@mcp.tool()
def lookup_order(order_id: int) -> str:
    """
    Look up an order's full event timeline (placed, shipped, delayed,
    delivered, etc.) and current status.

    Use this whenever the customer asks about an order's status,
    tracking, or history, and has NOT asked to cancel or refund it.

    Args:
        order_id: The numeric order ID, e.g. 7 for "order #7".
    """
    timeline = _lookup_order(order_id)
    if timeline is None:
        return f"No order found with ID #{order_id}. Ask the customer to double check the number."
    return timeline


@mcp.tool()
def cancel_order(order_id: int, confirm: bool) -> str:
    """
    Cancel an order, if it is still in a cancellable state.

    SAFETY: this performs a real, irreversible state change. Do not call
    this with confirm=True until the customer has explicitly confirmed
    in the conversation that they want to cancel this specific order.
    If you have not yet confirmed with the customer, call this with
    confirm=False first to see the confirmation prompt you should relay
    to them.

    Args:
        order_id: The numeric order ID to cancel.
        confirm: Must be True to actually perform the cancellation.
                 Pass False to get the confirmation question to ask first.
    """
    if not confirm:
        return (
            f"Before cancelling order #{order_id}, confirm with the customer: "
            f"\"Just to confirm -- you'd like to cancel order #{order_id}? "
            f"This can't be undone.\" Call this tool again with confirm=True "
            f"only after they explicitly agree."
        )
    result = _cancel_order(order_id)
    return result["reason"]


@mcp.tool()
def initiate_refund(order_id: int, damage_confirmed: bool, confirm: bool) -> str:
    """
    Issue a refund for an order, if it is eligible.

    Refunds are only eligible once an item has been returned and
    received. Orders over $150 are automatically flagged for manual
    human review rather than auto-approved -- this tool will report
    that instead of processing the refund in that case.

    SAFETY: this performs a real, irreversible state change and moves
    real money. Do not call this with confirm=True until the customer
    has (1) confirmed the returned item was undamaged, and (2)
    separately confirmed they want the refund processed.

    Args:
        order_id: The numeric order ID to refund.
        damage_confirmed: True only if the customer has confirmed the
                           returned item was undamaged.
        confirm: True only if the customer has explicitly confirmed they
                 want the refund processed, after the damage question.
    """
    if not damage_confirmed:
        return (
            f"Before processing a refund for order #{order_id}, ask the customer: "
            f"\"Can you confirm the returned item arrived at our warehouse with no damage?\" "
            f"Call this tool again with damage_confirmed=True only if they say yes."
        )
    if not confirm:
        return (
            f"Damage has been confirmed as none. Now ask: \"Just to confirm -- process "
            f"the refund for order #{order_id}?\" Call this tool again with confirm=True "
            f"only after they explicitly agree."
        )
    result = _initiate_refund(order_id)
    return result["reason"]


if __name__ == "__main__":
    mcp.run()