"""Pure helpers for the dashboard -- deliberately separated from Streamlit
rendering (docker/dashboard/app.py) so this logic is unit-testable without
a running Streamlit session or real HTTP calls.
"""

from __future__ import annotations


def build_status_rows(node_health: dict) -> list:
    """node_health: {node_id: bool} -> sorted list of {"node_id", "online"}
    rows, so table row order is a guarantee rather than an accident of
    dict iteration order.
    """
    return [{"node_id": node_id, "online": online} for node_id, online in sorted(node_health.items())]
