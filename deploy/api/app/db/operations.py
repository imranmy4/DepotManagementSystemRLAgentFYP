"""Order and crane-move operations: the only code that writes orders, move_log and
container locations.

Order flow:
    create_inbound_order / create_outbound_order   -> queued
    start_order (worker picks it up)               -> in_progress
    plan_moves (crane instructions from the policy) -> move_log rows, status planned
    confirm_move (operator confirms each move, in order) -> locations updated; the last
                                                           move completes the order
    reject_order (before any move is confirmed)    -> rejected

Every function works inside the caller's transaction and flushes but never commits.
Timestamps come from the database clock (transaction start time).
Positions are (block, bay, row, tier) tuples, as in the env.
"""
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import iso6346
from app.db.models import (
    Container, ContainerLocation, MoveLog, MoveStatus, MoveType, Order, OrderStatus, OrderType,
    Slot, YardState,
)

Pos = tuple[int, int, int, int]


class OperationError(ValueError):
    """Request that breaks the order flow or yard physics; nothing is written."""


@dataclass(frozen=True)
class PlannedMove:
    """One crane instruction from the planner; the container is looked up from from_pos."""
    move_type: MoveType
    from_pos: Pos | None
    to_pos: Pos | None
    crane_travel: int = 0


@dataclass(frozen=True)
class YardContainer:
    container_no: str
    pos: Pos
    arrived_at: datetime     # completed_at of the container's latest inbound order


def db_now(session: Session) -> datetime:
    """Database time at the start of the current transaction."""
    return session.scalar(select(func.now()))


# ─── Reads ─────────────────────────────────────────────────────────────────────

def _pos_select():
    return select(ContainerLocation.container_no, Slot.block_id, Slot.bay, Slot.row, Slot.tier) \
        .join(Slot, Slot.slot_id == ContainerLocation.slot_id) \
        .where(ContainerLocation.removed_at.is_(None))


def occupancy(session: Session) -> dict[Pos, str]:
    """Current yard as {position: container_no}."""
    return {(b, ba, r, t): c for c, b, ba, r, t in session.execute(_pos_select())}


def current_yard(session: Session) -> list[YardContainer]:
    """Every container in the yard with its position and yard arrival time."""
    arrived = select(func.max(Order.completed_at)).where(
        Order.container_no == ContainerLocation.container_no,
        Order.type == OrderType.INBOUND, Order.status == OrderStatus.DONE,
    ).scalar_subquery()
    rows = session.execute(_pos_select().add_columns(arrived))
    return [YardContainer(c, (b, ba, r, t), at) for c, b, ba, r, t, at in rows]


def crane_position(session: Session) -> tuple[int, int, int]:
    """Crane (block, bay, row)."""
    ys = session.get(YardState, 1)
    if ys is None:
        raise OperationError("yard_state missing: run the seed command")
    return ys.crane_block, ys.crane_bay, ys.crane_row


def order_moves(session: Session, order_id: int) -> list[MoveLog]:
    return list(session.scalars(
        select(MoveLog).where(MoveLog.order_id == order_id).order_by(MoveLog.seq)))


def _slot_id(session: Session, pos: Pos) -> int:
    sid = session.scalar(select(Slot.slot_id).where(
        Slot.block_id == pos[0], Slot.bay == pos[1], Slot.row == pos[2], Slot.tier == pos[3]))
    if sid is None:
        raise OperationError(f"no slot at {pos}")
    return sid


def _slot_pos(session: Session, slot_id: int) -> Pos:
    s = session.get(Slot, slot_id)
    return s.block_id, s.bay, s.row, s.tier


def _lock_order(session: Session, order_id: int) -> Order:
    order = session.get(Order, order_id, with_for_update=True)
    if order is None:
        raise OperationError(f"order {order_id} not found")
    return order


# ─── Yard physics ──────────────────────────────────────────────────────────────

def _check_move(occupied: dict[Pos, str], src: Pos | None, dst: Pos | None) -> None:
    """Raise unless the move takes a top container and lands it on a stack, not in mid-air."""
    if src is not None:
        if src not in occupied:
            raise OperationError(f"no container at {src}")
        if (*src[:3], src[3] + 1) in occupied:
            raise OperationError(f"container at {src} is not on top of its stack")
    if dst is not None:
        if dst in occupied:
            raise OperationError(f"slot {dst} is occupied")
        if src is not None and src[:3] == dst[:3]:
            raise OperationError(f"move {src} -> {dst} stays in the same stack")
        if dst[3] > 0 and (*dst[:3], dst[3] - 1) not in occupied:
            raise OperationError(f"slot {dst} has nothing below it")


def _apply(occupied: dict[Pos, str], src: Pos | None, dst: Pos | None, container_no: str):
    if src is not None:
        del occupied[src]
    if dst is not None:
        occupied[dst] = container_no


# ─── Orders ────────────────────────────────────────────────────────────────────

def create_inbound_order(session: Session, container_no: str, truck_no: str | None = None,
                         owner_line: str | None = None) -> Order:
    """Queue an inbound order; registers the container on its first visit."""
    if not iso6346.is_valid(container_no):
        raise OperationError(f"{container_no!r} is not a valid ISO 6346 container number")
    if session.get(Container, container_no) is None:
        session.add(Container(container_no=container_no, owner_line=owner_line))
    in_yard = session.scalar(select(ContainerLocation.id).where(
        ContainerLocation.container_no == container_no, ContainerLocation.removed_at.is_(None)))
    if in_yard is not None:
        raise OperationError(f"{container_no} is already in the yard")
    open_order = session.scalar(select(Order.order_id).where(
        Order.container_no == container_no, Order.type == OrderType.INBOUND,
        Order.status.in_([OrderStatus.QUEUED, OrderStatus.IN_PROGRESS])))
    if open_order is not None:
        raise OperationError(f"{container_no} already has open inbound order {open_order}")
    order = Order(type=OrderType.INBOUND, container_no=container_no, truck_no=truck_no)
    session.add(order)
    session.flush()
    return order


def create_outbound_order(session: Session, truck_no: str | None = None) -> Order:
    """Queue an outbound order; the container is chosen later by the policy."""
    order = Order(type=OrderType.OUTBOUND, truck_no=truck_no)
    session.add(order)
    session.flush()
    return order


def start_order(session: Session, order_id: int, decided_by: str) -> Order:
    """Mark a queued order as being processed by `decided_by` (the deployed model's label)."""
    order = _lock_order(session, order_id)
    if order.status != OrderStatus.QUEUED:
        raise OperationError(f"order {order_id} is {order.status.value}, not queued")
    order.status = OrderStatus.IN_PROGRESS
    order.decided_by = decided_by
    session.flush()
    return order


def plan_moves(session: Session, order_id: int, moves: list[PlannedMove]) -> list[MoveLog]:
    """Store the crane instructions of an in-progress order as planned moves.

    Inbound: exactly one store. Outbound: relocations, then one retrieve. The whole sequence is
    checked against the current yard; an outbound order gets the retrieved container's number.
    """
    order = _lock_order(session, order_id)
    if order.status != OrderStatus.IN_PROGRESS:
        raise OperationError(f"order {order_id} is {order.status.value}, not in_progress")
    if order_moves(session, order_id):
        raise OperationError(f"order {order_id} already has moves")

    types = [m.move_type for m in moves]
    if order.type == OrderType.INBOUND and types != [MoveType.STORE]:
        raise OperationError("an inbound order needs exactly one store move")
    if order.type == OrderType.OUTBOUND and (
            not types or types[-1] != MoveType.RETRIEVE
            or any(t != MoveType.RELOCATE for t in types[:-1])):
        raise OperationError("an outbound order needs relocations followed by one retrieve")

    occupied = occupancy(session)
    rows = []
    for seq, m in enumerate(moves):
        _check_move(occupied, m.from_pos, m.to_pos)
        cno = order.container_no if m.move_type == MoveType.STORE else occupied[m.from_pos]
        _apply(occupied, m.from_pos, m.to_pos, cno)
        rows.append(MoveLog(
            order_id=order_id, seq=seq, move_type=m.move_type, container_no=cno,
            from_slot_id=_slot_id(session, m.from_pos) if m.from_pos else None,
            to_slot_id=_slot_id(session, m.to_pos) if m.to_pos else None,
            crane_travel=m.crane_travel))

    if order.type == OrderType.OUTBOUND:
        order.container_no = rows[-1].container_no
    session.add_all(rows)
    session.flush()
    return rows


def confirm_move(session: Session, move_id: int) -> MoveLog:
    """Record that the crane carried out a planned move; moves must be confirmed in order.

    Updates container locations (and the crane after a retrieve); the last move completes
    the order.
    """
    now = db_now(session)
    move = session.get(MoveLog, move_id, with_for_update=True)
    if move is None:
        raise OperationError(f"move {move_id} not found")
    order = _lock_order(session, move.order_id)
    if order.status != OrderStatus.IN_PROGRESS:
        raise OperationError(f"order {order.order_id} is {order.status.value}, not in_progress")
    if move.status != MoveStatus.PLANNED:
        raise OperationError(f"move {move_id} is already done")
    moves = order_moves(session, order.order_id)
    if any(m.status != MoveStatus.DONE for m in moves if m.seq < move.seq):
        raise OperationError(f"move {move_id} confirmed before the earlier moves of its order")

    src = _slot_pos(session, move.from_slot_id) if move.from_slot_id else None
    dst = _slot_pos(session, move.to_slot_id) if move.to_slot_id else None
    occupied = occupancy(session)
    _check_move(occupied, src, dst)
    if src is not None and occupied[src] != move.container_no:
        raise OperationError(f"expected {move.container_no} at {src}, found {occupied[src]}")

    if src is not None:
        loc = session.scalar(select(ContainerLocation).where(
            ContainerLocation.container_no == move.container_no,
            ContainerLocation.removed_at.is_(None)))
        loc.removed_at = now
        session.flush()          # free the slot before a new location may take it
    if dst is not None:
        session.add(ContainerLocation(container_no=move.container_no,
                                      slot_id=move.to_slot_id, placed_at=now))
    if move.move_type == MoveType.RETRIEVE:
        ys = session.get(YardState, 1)
        ys.crane_block, ys.crane_bay, ys.crane_row = src[:3]

    move.status = MoveStatus.DONE
    move.done_at = now
    if move.seq == moves[-1].seq:
        order.status = OrderStatus.DONE
        order.completed_at = now
    session.flush()
    return move


def reject_order(session: Session, order_id: int) -> Order:
    """Reject an order that has no confirmed moves; its planned moves are deleted."""
    order = _lock_order(session, order_id)
    if order.status not in (OrderStatus.QUEUED, OrderStatus.IN_PROGRESS):
        raise OperationError(f"order {order_id} is {order.status.value}")
    moves = order_moves(session, order_id)
    if any(m.status == MoveStatus.DONE for m in moves):
        raise OperationError(f"order {order_id} already has confirmed moves")
    for m in moves:
        session.delete(m)
    order.status = OrderStatus.REJECTED
    if order.type == OrderType.OUTBOUND:
        order.container_no = None
    session.flush()
    return order
