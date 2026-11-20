"""Crane move confirmation: the operator reports a move as done."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import operations
from app.db.models import Order
from app.deps import get_session
from app.routers.orders import order_detail
from app.schemas import OrderDetailOut

router = APIRouter(prefix="/moves", tags=["moves"])


@router.post("/{move_id}/confirm", response_model=OrderDetailOut)
def confirm(move_id: int, session: Session = Depends(get_session)):
    """Confirm one move (moves of an order in sequence); returns its order with the updated
    moves. The last move completes the order and frees the crane for the next one."""
    move = operations.confirm_move(session, move_id)
    session.commit()
    return order_detail(session, session.get(Order, move.order_id))
