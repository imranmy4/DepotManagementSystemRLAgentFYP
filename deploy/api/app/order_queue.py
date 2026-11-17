"""Redis order queue. Redis holds order ids only; Postgres stays the source of truth, so the
queue can always be rebuilt from the orders table.

Two FIFO lists, one per order type, and a processing list for the order the worker holds:
    orders:outbound, orders:inbound   new ids pushed left, taken from the right (oldest first)
    orders:processing                 at most one id while the worker plans it
Outbound orders are taken before inbound ones, as in training.
"""
import redis
from sqlalchemy.orm import Session

from app.db.models import OrderType
from app.db.operations import queued_orders
from app.settings import settings

OUTBOUND = "orders:outbound"
INBOUND = "orders:inbound"
PROCESSING = "orders:processing"


def connect() -> redis.Redis:
    return redis.Redis.from_url(settings.redis_url, decode_responses=True)


def _key(order_type: OrderType) -> str:
    return OUTBOUND if order_type == OrderType.OUTBOUND else INBOUND


def submit(r: redis.Redis, order_id: int, order_type: OrderType) -> None:
    """Queue an order. Call only after the order's transaction has committed."""
    r.lpush(_key(order_type), order_id)


def take_next(r: redis.Redis, yard_has_container: bool) -> int | None:
    """Move the next order id into the processing list and return it, or None if none waits.

    Outbound first; while the yard is empty, inbound goes first so the outbound can be served,
    and an outbound with no inbound waiting is still returned (the worker rejects it).
    """
    order = [OUTBOUND, INBOUND] if yard_has_container else [INBOUND, OUTBOUND]
    for key in order:
        order_id = r.lmove(key, PROCESSING, "RIGHT", "LEFT")
        if order_id is not None:
            return int(order_id)
    return None


def finish(r: redis.Redis, order_id: int) -> None:
    """Drop an id from the processing list once its order is planned or rejected in Postgres."""
    r.lrem(PROCESSING, 0, order_id)


def reconcile(r: redis.Redis, session: Session) -> int:
    """Make the lists match the queued orders in Postgres; returns how many ids were re-pushed.

    Queued orders missing from the lists (worker stopped mid-order, lost Redis data, or a crash
    between commit and submit) are put back at the front, oldest first; the processing list is
    then cleared, since an id there whose order is no longer queued is finished. Ids are
    only ever added, so an order submitted concurrently is never lost; a duplicate is harmless
    because the worker skips ids whose order is no longer queued.
    """
    held = {int(x) for x in r.lrange(PROCESSING, 0, -1)}
    present = {int(x) for key in (OUTBOUND, INBOUND) for x in r.lrange(key, 0, -1)}
    missing = [o for o in queued_orders(session) if o.order_id not in present]

    # RPUSH puts ids at the taking end, so push newest first to leave the oldest next in line.
    for order in reversed(missing):
        r.rpush(_key(order.type), order.order_id)
    for order_id in held:
        r.lrem(PROCESSING, 0, order_id)
    return len(missing)
