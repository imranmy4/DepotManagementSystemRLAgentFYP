"""Tests for the Redis order queue: FIFO per type, outbound first, and rebuilding from Postgres."""
from app import order_queue as q
from app.db.models import OrderType
from app.db.operations import create_inbound_order, create_outbound_order
from helpers import container_no

IN, OUT = OrderType.INBOUND, OrderType.OUTBOUND


def _drain(r, yard_has_container):
    taken = []
    while (order_id := q.take_next(r, yard_has_container)) is not None:
        taken.append(order_id)
        q.finish(r, order_id)
    return taken


def test_outbound_first_then_fifo(r):
    for order_id, kind in [(1, IN), (2, OUT), (3, IN), (4, OUT)]:
        q.submit(r, order_id, kind)
    assert _drain(r, yard_has_container=True) == [2, 4, 1, 3]


def test_empty_yard_takes_inbound_first(r):
    for order_id, kind in [(1, OUT), (2, IN), (3, IN)]:
        q.submit(r, order_id, kind)
    assert _drain(r, yard_has_container=False) == [2, 3, 1]


def test_take_holds_id_in_processing(r):
    q.submit(r, 7, IN)
    assert q.take_next(r, True) == 7
    assert r.lrange(q.PROCESSING, 0, -1) == ["7"]
    q.finish(r, 7)
    assert r.llen(q.PROCESSING) == 0


def test_reconcile_restores_lost_orders_in_order(r, yard):
    import random
    rng = random.Random(0)
    a = create_inbound_order(yard, container_no(rng))
    b = create_outbound_order(yard)
    c = create_inbound_order(yard, container_no(rng))
    q.submit(r, c.order_id, IN)             # a and b were never pushed (lost)
    assert q.reconcile(r, yard) == 2
    assert _drain(r, yard_has_container=True) == [b.order_id, a.order_id, c.order_id]


def test_reconcile_returns_held_order_and_drops_finished(r, yard):
    import random
    order = create_inbound_order(yard, container_no(random.Random(1)))
    q.submit(r, order.order_id, IN)
    assert q.take_next(r, True) == order.order_id           # worker stopped mid-order
    r.lpush(q.PROCESSING, 999)                              # stale id of a finished order
    q.reconcile(r, yard)
    assert r.llen(q.PROCESSING) == 0
    assert _drain(r, True) == [order.order_id]
