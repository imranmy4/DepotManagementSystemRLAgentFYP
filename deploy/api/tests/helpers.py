"""Test-only helpers: a small yard, container number generation and simulated time."""
import random

from sqlalchemy import text

from app.iso6346 import check_digit
from depot.env import DepotConfig

# Small yard: 2 blocks side by side, 2 bays x 2 rows x 3 tiers each.
TINY = DepotConfig(n_blocks=2, block_rows=1, n_bays=2, n_rows=2, n_tiers=3)

TIME_COLUMNS = {
    "container": ["created_at"],
    "container_location": ["placed_at", "removed_at"],
    "orders": ["created_at", "completed_at"],
    "move_log": ["created_at", "done_at"],
}


def container_no(rng: random.Random, owner: str = "TSTU") -> str:
    """Random valid ISO 6346 number."""
    first10 = f"{owner}{rng.randrange(1_000_000):06d}"
    return first10 + str(check_digit(first10))


def pass_days(session, days: float) -> None:
    """Simulate time passing by moving every stored timestamp `days` into the past."""
    for table, cols in TIME_COLUMNS.items():
        sets = ", ".join(f"{c} = {c} - make_interval(secs => :s)" for c in cols)
        session.execute(text(f"UPDATE {table} SET {sets}"), {"s": days * 86400})
    session.expire_all()
