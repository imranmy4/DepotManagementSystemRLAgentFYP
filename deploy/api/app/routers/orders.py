"""Order endpoints: gate-in / gate-out requests, order status and rejection."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app import order_queue
from app.db import operations
from app.db.models import Order, OrderStatus
from app.deps import get_redis, get_session
from app.schemas import (
    InboundOrderIn, MoveOut, OrderDetailOut, OrderOut, OutboundOrderIn, Position, RejectIn,
)

router = APIRouter(prefix="/orders", tags=["orders"])


def _pos(p) -> Position | None:
    return None if p is None else Position(block=p[0], bay=p[1], row=p[2], tier=p[3])


def order_detail(session: Session, order: Order) -> OrderDetailOut:
    """An order with its crane moves, slot ids resolved to positions."""
    slots = operations.slot_positions(session)
    moves = [MoveOut(move_id=m.move_id, seq=m.seq, status=m.status.value,
                     move_type=m.move_type.value, container_no=m.container_no,
                     from_pos=_pos(slots.get(m.from_slot_id)), to_pos=_pos(slots.get(m.to_slot_id)),
                     crane_travel=m.crane_travel, done_at=m.done_at)
             for m in operations.order_moves(session, order.order_id)]
    return OrderDetailOut(**OrderOut.model_validate(order).model_dump(), moves=moves)


@router.post("/inbound", response_model=OrderOut, status_code=status.HTTP_201_CREATED)
def create_inbound(body: InboundOrderIn, session: Session = Depends(get_session),
                   r=Depends(get_redis)):
    """Register an arriving container; the worker decides where it goes."""
    order = operations.create_inbound_order(session, body.container_no, body.truck_no,
                                            body.owner_line)
    session.commit()    # persist before queueing
    order_queue.submit(r, order.order_id, order.type)
    return order


@router.post("/outbound", response_model=OrderOut, status_code=status.HTTP_201_CREATED)
def create_outbound(body: OutboundOrderIn, session: Session = Depends(get_session),
                    r=Depends(get_redis)):
    """Request one container for a truck; the worker decides which one is released."""
    order = operations.create_outbound_order(session, body.truck_no)
    session.commit()
    order_queue.submit(r, order.order_id, order.type)
    return order


@router.get("", response_model=list[OrderOut])
def list_orders(status: OrderStatus | None = None, limit: int = 50,
                session: Session = Depends(get_session)):
    """Latest orders first, optionally filtered by status."""
    return operations.recent_orders(session, min(max(limit, 1), 500), status)


@router.get("/current", response_model=OrderDetailOut | None)
def current_order(session: Session = Depends(get_session)):
    """The order the crane is working on, with its moves; null when the crane is free."""
    order = operations.order_in_progress(session)
    return None if order is None else order_detail(session, order)


@router.get("/{order_id}", response_model=OrderDetailOut)
def get_order(order_id: int, session: Session = Depends(get_session)):
    order = session.get(Order, order_id)
    if order is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"order {order_id} not found")
    return order_detail(session, order)


@router.post("/{order_id}/reject", response_model=OrderOut)
def reject(order_id: int, body: RejectIn, session: Session = Depends(get_session)):
    """Cancel an order before any of its moves is confirmed. A queued id left in Redis is
    skipped by the worker, which checks the order status."""
    order = operations.reject_order(session, order_id, body.reason)
    session.commit()
    return order
