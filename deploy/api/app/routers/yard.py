"""Yard view: layout, crane, every container with its dwell, and the queue size."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import operations
from app.deps import get_session
from app.schemas import BlockOut, Position, YardContainerOut, YardOut
from app.settings import settings

router = APIRouter(prefix="/yard", tags=["yard"])


@router.get("", response_model=YardOut)
def yard(session: Session = Depends(get_session)):
    now = operations.db_now(session)
    cb, cba, cr = operations.crane_position(session)
    containers = [
        YardContainerOut(
            container_no=c.container_no,
            pos=Position(block=c.pos[0], bay=c.pos[1], row=c.pos[2], tier=c.pos[3]),
            arrived_at=c.arrived_at,
            dwell_ticks=0.0 if c.arrived_at is None
            else max(0.0, (now - c.arrived_at).total_seconds() / settings.tick_seconds))
        for c in operations.current_yard(session)]
    current = operations.order_in_progress(session)
    return YardOut(
        now=now, tick_seconds=settings.tick_seconds,
        blocks=[BlockOut.model_validate(b) for b in operations.yard_blocks(session)],
        crane=Position(block=cb, bay=cba, row=cr, tier=0),
        containers=containers,
        queued=len(operations.queued_orders(session)),
        in_progress=None if current is None else current.order_id)
