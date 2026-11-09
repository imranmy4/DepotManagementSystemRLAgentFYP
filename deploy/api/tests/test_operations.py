"""Tests for the order flow: inbound store, outbound with relocations, validation and the
database integrity rules behind it."""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.exc import IntegrityError

from app.db.models import ContainerLocation, MoveStatus, MoveType, OrderStatus
from app.db.operations import (
    OperationError, PlannedMove, confirm_move, crane_position, create_inbound_order,
    create_outbound_order, current_yard, occupancy, order_moves, plan_moves, reject_order,
    start_order,
)

T0 = datetime(2026, 11, 1, 8, 0, tzinfo=timezone.utc)
MODEL = "run_test/best_model"


def store(s, cno, pos, now=T0):
    """Run a full inbound order that places `cno` at `pos`."""
    order = create_inbound_order(s, cno, now=now)
    start_order(s, order.order_id, MODEL)
    (move,) = plan_moves(s, order.order_id, [PlannedMove(MoveType.STORE, None, pos)], now=now)
    confirm_move(s, move.move_id, now=now)
    return order


def test_inbound_stores_container(yard):
    order = store(yard, "ABCU1234565", (0, 0, 0, 0))
    assert order.status == OrderStatus.DONE and order.completed_at == T0
    assert order.decided_by == MODEL
    assert occupancy(yard) == {(0, 0, 0, 0): "ABCU1234565"}
    (c,) = current_yard(yard)
    assert c.arrived_at == T0


def test_inbound_rejects_container_already_in_yard(yard):
    store(yard, "ABCU1234565", (0, 0, 0, 0))
    with pytest.raises(OperationError, match="already in the yard"):
        create_inbound_order(yard, "ABCU1234565")


def test_store_cannot_float(yard):
    order = create_inbound_order(yard, "ABCU1234565")
    start_order(yard, order.order_id, MODEL)
    with pytest.raises(OperationError, match="nothing below"):
        plan_moves(yard, order.order_id, [PlannedMove(MoveType.STORE, None, (0, 0, 0, 1))])


def test_plan_requires_started_order(yard):
    order = create_inbound_order(yard, "ABCU1234565")
    with pytest.raises(OperationError, match="not in_progress"):
        plan_moves(yard, order.order_id, [PlannedMove(MoveType.STORE, None, (0, 0, 0, 0))])


def _outbound_with_dig(yard):
    """Bottom container A (older) under B; outbound digs out A by relocating B."""
    store(yard, "AAAU0000001", (0, 0, 0, 0), now=T0)
    store(yard, "BBBU0000002", (0, 0, 0, 1), now=T0 + timedelta(days=1))
    order = create_outbound_order(yard)
    start_order(yard, order.order_id, MODEL)
    moves = plan_moves(yard, order.order_id, [
        PlannedMove(MoveType.RELOCATE, (0, 0, 0, 1), (1, 1, 1, 0)),
        PlannedMove(MoveType.RETRIEVE, (0, 0, 0, 0), None, crane_travel=1),
    ])
    return order, moves


def test_outbound_with_relocation(yard):
    order, (reloc, retrieve) = _outbound_with_dig(yard)
    assert order.container_no == "AAAU0000001"
    assert (reloc.container_no, retrieve.container_no) == ("BBBU0000002", "AAAU0000001")

    later = T0 + timedelta(days=5)
    confirm_move(yard, reloc.move_id, now=later)
    assert order.status == OrderStatus.IN_PROGRESS
    confirm_move(yard, retrieve.move_id, now=later)

    assert order.status == OrderStatus.DONE
    assert occupancy(yard) == {(1, 1, 1, 0): "BBBU0000002"}
    assert crane_position(yard) == (0, 0, 0)
    # A relocated container keeps its yard arrival time.
    (b,) = current_yard(yard)
    assert b.arrived_at == T0 + timedelta(days=1)
    assert all(m.status == MoveStatus.DONE for m in order_moves(yard, order.order_id))


def test_moves_confirm_in_order(yard):
    _, (_, retrieve) = _outbound_with_dig(yard)
    with pytest.raises(OperationError, match="before the earlier moves"):
        confirm_move(yard, retrieve.move_id)


def test_retrieve_must_take_top_container(yard):
    store(yard, "AAAU0000001", (0, 0, 0, 0))
    store(yard, "BBBU0000002", (0, 0, 0, 1))
    order = create_outbound_order(yard)
    start_order(yard, order.order_id, MODEL)
    with pytest.raises(OperationError, match="not on top"):
        plan_moves(yard, order.order_id, [PlannedMove(MoveType.RETRIEVE, (0, 0, 0, 0), None)])


def test_outbound_must_end_with_retrieve(yard):
    store(yard, "AAAU0000001", (0, 0, 0, 0))
    order = create_outbound_order(yard)
    start_order(yard, order.order_id, MODEL)
    with pytest.raises(OperationError, match="followed by one retrieve"):
        plan_moves(yard, order.order_id, [PlannedMove(MoveType.RELOCATE, (0, 0, 0, 0), (1, 0, 0, 0))])


def test_reject_before_confirm(yard):
    order, _ = _outbound_with_dig(yard)
    reject_order(yard, order.order_id)
    assert order.status == OrderStatus.REJECTED and order.container_no is None
    assert order_moves(yard, order.order_id) == []
    assert len(occupancy(yard)) == 2


def test_reject_after_confirm_refused(yard):
    order, (reloc, _) = _outbound_with_dig(yard)
    confirm_move(yard, reloc.move_id)
    with pytest.raises(OperationError, match="confirmed moves"):
        reject_order(yard, order.order_id)


def test_db_refuses_double_booked_slot(yard):
    store(yard, "AAAU0000001", (0, 0, 0, 0))
    slot_id = yard.query(ContainerLocation).one().slot_id
    yard.add(ContainerLocation(container_no="AAAU0000001", slot_id=slot_id))
    with pytest.raises(IntegrityError):
        yard.flush()
