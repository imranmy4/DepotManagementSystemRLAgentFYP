"""Request and response bodies of the API."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class InboundOrderIn(BaseModel):
    container_no: str = Field(examples=["CSQU3054383"])
    truck_no: str | None = None
    owner_line: str | None = None


class OutboundOrderIn(BaseModel):
    truck_no: str | None = None


class RejectIn(BaseModel):
    reason: str = Field(min_length=1, max_length=200)


class OrderOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    order_id: int
    type: str
    status: str
    container_no: str | None
    truck_no: str | None
    decided_by: str | None
    reject_reason: str | None
    created_at: datetime
    completed_at: datetime | None


class Position(BaseModel):
    block: int
    bay: int
    row: int
    tier: int


class MoveOut(BaseModel):
    """One crane instruction; from is empty for a store, to is empty for a retrieve."""
    move_id: int
    seq: int
    status: str
    move_type: str
    container_no: str
    from_pos: Position | None
    to_pos: Position | None
    crane_travel: int
    done_at: datetime | None


class OrderDetailOut(OrderOut):
    moves: list[MoveOut]


class BlockOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    block_id: int
    num_bays: int
    num_rows: int
    num_tiers: int
    grid_row: int
    grid_col: int


class YardContainerOut(BaseModel):
    container_no: str
    pos: Position
    arrived_at: datetime | None
    dwell_ticks: float          # time in the yard in model ticks (days at the default tick)


class YardOut(BaseModel):
    now: datetime
    tick_seconds: int
    blocks: list[BlockOut]
    crane: Position             # tier is always 0: the crane is over a stack
    containers: list[YardContainerOut]
    queued: int
    in_progress: int | None     # order id the crane is working on
