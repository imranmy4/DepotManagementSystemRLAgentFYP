"""Order worker: the single process that decides orders with the deployed model.

    python -m app.worker        (from deploy/api)

Exactly one worker runs (one crane). It plans one order at a time and only when the crane is
free (no order in progress), so every plan is made on the real, current yard.
"""
import logging
import time

import redis

from app import order_queue
from app.adapter import NoValidAction, decide, load_policy
from app.db.models import Order, OrderStatus
from app.db.operations import (
    container_count, order_in_progress, plan_moves, reject_order, start_order,
)
from app.db.seed import model_source
from app.db.session import SessionLocal

log = logging.getLogger("depot.worker")

POLL_SECONDS = 1.0


def process(session, policy, order_id: int, decided_by: str) -> None:
    """Plan one queued order (or reject it if it cannot be served) inside `session`'s transaction."""
    order = session.get(Order, order_id, with_for_update=True)
    if order is None or order.status != OrderStatus.QUEUED:
        log.info("order %s skipped (no longer queued)", order_id)
        return
    start_order(session, order_id, decided_by)
    try:
        moves = decide(session, policy, order)
    except NoValidAction as e:
        reject_order(session, order_id, str(e))
        log.info("order %s rejected: %s", order_id, e)
        return
    plan_moves(session, order_id, moves)
    log.info("order %s (%s) planned: %d move(s)", order_id, order.type.value, len(moves))


def step(r: redis.Redis, policy, decided_by: str) -> bool:
    """Take and plan the next order if the crane is free. Returns True if an order was handled."""
    with SessionLocal.begin() as s:
        if order_in_progress(s) is not None:
            return False
        has_container = container_count(s) > 0

    order_id = order_queue.take_next(r, has_container)
    if order_id is None:
        with SessionLocal.begin() as s:
            if order_queue.reconcile(r, s):
                log.warning("re-queued orders missing from Redis")
        return False

    try:
        with SessionLocal.begin() as s:
            process(s, policy, order_id, decided_by)
    except Exception:
        # A failed plan is rolled back; reject the order so it cannot block the queue.
        log.exception("order %s: planning failed", order_id)
        with SessionLocal.begin() as s:
            reject_order(s, order_id, "planning failed (see worker log)")
    order_queue.finish(r, order_id)
    return True


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    with SessionLocal() as s:
        policy = load_policy(s)
    decided_by = model_source()
    r = order_queue.connect()
    with SessionLocal.begin() as s:
        n = order_queue.reconcile(r, s)
    log.info("worker started with model %s (%d order(s) re-queued)", decided_by, n)

    while True:
        if not step(r, policy, decided_by):
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
