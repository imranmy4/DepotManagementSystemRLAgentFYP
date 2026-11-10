"""Tests for the worker: one order in flight, outbound first, rejections and recovery."""
import random

from app import order_queue as q
from app.db.models import Order, OrderStatus, OrderType
from app.db.operations import (
    confirm_move, create_inbound_order, create_outbound_order, order_moves, reject_order,
)
from app.worker import step
from helpers import StubPolicy, container_no

MODEL = "run_test/best_model"


def submit_inbound(Session, r, cno):
    with Session.begin() as s:
        order = create_inbound_order(s, cno)
    q.submit(r, order.order_id, OrderType.INBOUND)
    return order.order_id


def submit_outbound(Session, r):
    with Session.begin() as s:
        order = create_outbound_order(s)
    q.submit(r, order.order_id, OrderType.OUTBOUND)
    return order.order_id


def status(Session, order_id):
    with Session() as s:
        return s.get(Order, order_id).status


def confirm_all(Session, order_id):
    with Session.begin() as s:
        for m in order_moves(s, order_id):
            confirm_move(s, m.move_id)


def test_plans_one_order_at_a_time(committed, r):
    rng = random.Random(0)
    first = submit_inbound(committed, r, container_no(rng))
    second = submit_inbound(committed, r, container_no(rng))

    assert step(r, StubPolicy(), MODEL) is True
    assert status(committed, first) == OrderStatus.IN_PROGRESS
    assert step(r, StubPolicy(), MODEL) is False           # crane busy with the first order
    assert status(committed, second) == OrderStatus.QUEUED

    confirm_all(committed, first)
    assert status(committed, first) == OrderStatus.DONE
    assert step(r, StubPolicy(), MODEL) is True
    assert status(committed, second) == OrderStatus.IN_PROGRESS
    with committed() as s:
        assert s.get(Order, second).decided_by == MODEL
    assert r.llen(q.PROCESSING) == 0


def test_outbound_goes_before_earlier_inbound(committed, r):
    rng = random.Random(1)
    stored = submit_inbound(committed, r, container_no(rng))
    step(r, StubPolicy(), MODEL)
    confirm_all(committed, stored)

    inbound = submit_inbound(committed, r, container_no(rng))
    outbound = submit_outbound(committed, r)
    step(r, StubPolicy(), MODEL)
    assert status(committed, outbound) == OrderStatus.IN_PROGRESS
    assert status(committed, inbound) == OrderStatus.QUEUED


def test_outbound_on_empty_yard_rejected(committed, r):
    order_id = submit_outbound(committed, r)
    assert step(r, StubPolicy(), MODEL) is True
    with committed() as s:
        order = s.get(Order, order_id)
        assert order.status == OrderStatus.REJECTED
        assert order.reject_reason == "no container in the yard"


def test_cancelled_order_is_skipped(committed, r):
    order_id = submit_inbound(committed, r, container_no(random.Random(2)))
    with committed.begin() as s:
        reject_order(s, order_id, "truck left")
    assert step(r, StubPolicy(), MODEL) is True
    with committed() as s:
        assert s.get(Order, order_id).reject_reason == "truck left"
        assert order_moves(s, order_id) == []


def test_planning_failure_rejects_order(committed, r):
    class Broken(StubPolicy):
        def act(self, env):
            raise RuntimeError("model crashed")

    order_id = submit_inbound(committed, r, container_no(random.Random(3)))
    assert step(r, Broken(), MODEL) is True
    with committed() as s:
        order = s.get(Order, order_id)
        assert order.status == OrderStatus.REJECTED
        assert order.reject_reason.startswith("planning failed")
    assert r.llen(q.PROCESSING) == 0


def test_idle_worker_requeues_lost_order(committed, r):
    with committed.begin() as s:                            # committed but never submitted
        order_id = create_inbound_order(s, container_no(random.Random(4))).order_id
    assert step(r, StubPolicy(), MODEL) is False            # idle pass re-queues it
    assert step(r, StubPolicy(), MODEL) is True
    assert status(committed, order_id) == OrderStatus.IN_PROGRESS
