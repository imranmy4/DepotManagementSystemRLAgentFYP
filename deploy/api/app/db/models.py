"""Database schema: yard layout, containers, locations, orders and crane moves."""
import enum
from datetime import datetime

from sqlalchemy import (
    BigInteger, CheckConstraint, DateTime, Enum, ForeignKey, Index, Integer, MetaData,
    SmallInteger, String, UniqueConstraint, func, text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Constraint naming pattern used by Alembic migrations.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def _enum(cls):
    """VARCHAR column with a CHECK constraint, storing the enum member values."""
    return Enum(cls, native_enum=False, create_constraint=True, length=20,
                values_callable=lambda e: [m.value for m in e])


TIMESTAMP = DateTime(timezone=True)


class OrderType(str, enum.Enum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class OrderStatus(str, enum.Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    REJECTED = "rejected"


class MoveType(str, enum.Enum):
    STORE = "store"         # gate -> slot
    RELOCATE = "relocate"   # slot -> slot
    RETRIEVE = "retrieve"   # slot -> gate


class MoveStatus(str, enum.Enum):
    PLANNED = "planned"     # issued, not yet confirmed
    DONE = "done"


# ─── Yard layout ───────────────────────────────────────────────────────────────

class YardBlock(Base):
    __tablename__ = "yard_block"
    block_id: Mapped[int] = mapped_column(SmallInteger, primary_key=True, autoincrement=False)
    num_bays: Mapped[int] = mapped_column(SmallInteger)
    num_rows: Mapped[int] = mapped_column(SmallInteger)
    num_tiers: Mapped[int] = mapped_column(SmallInteger)
    # Position on the block grid; crane distance and block adjacency are measured on it.
    grid_row: Mapped[int] = mapped_column(SmallInteger)
    grid_col: Mapped[int] = mapped_column(SmallInteger)
    __table_args__ = (
        CheckConstraint("block_id >= 0", name="block_id_nonneg"),
        CheckConstraint("num_bays > 0 AND num_rows > 0 AND num_tiers > 0", name="dims_positive"),
        CheckConstraint("grid_row >= 0 AND grid_col >= 0", name="grid_nonneg"),
        UniqueConstraint("grid_row", "grid_col", name="uq_block_grid_position"),
    )


class Slot(Base):
    __tablename__ = "slot"
    slot_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    block_id: Mapped[int] = mapped_column(ForeignKey("yard_block.block_id"))
    bay: Mapped[int] = mapped_column(SmallInteger)
    row: Mapped[int] = mapped_column(SmallInteger)
    tier: Mapped[int] = mapped_column(SmallInteger)
    __table_args__ = (
        UniqueConstraint("block_id", "bay", "row", "tier", name="uq_slot_position"),
        CheckConstraint("bay >= 0 AND row >= 0 AND tier >= 0", name="coords_nonneg"),
    )


class YardState(Base):
    """Single row holding the live crane position."""
    __tablename__ = "yard_state"
    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True, default=1)
    crane_block: Mapped[int] = mapped_column(ForeignKey("yard_block.block_id"))
    crane_bay: Mapped[int] = mapped_column(SmallInteger)
    crane_row: Mapped[int] = mapped_column(SmallInteger)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now(),
                                                 onupdate=func.now())
    __table_args__ = (CheckConstraint("id = 1", name="single_row"),)


# ─── Containers and where they are ─────────────────────────────────────────────

class Container(Base):
    """A physical container identified by its ISO 6346 number."""
    __tablename__ = "container"
    container_no: Mapped[str] = mapped_column(String(11), primary_key=True)  # ISO 6346
    size_ft: Mapped[int] = mapped_column(SmallInteger, server_default=text("20"))
    type: Mapped[str] = mapped_column(String(4), server_default=text("'GP'"))
    owner_line: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now())
    __table_args__ = (
        CheckConstraint("container_no ~ '^[A-Z]{4}[0-9]{7}$'", name="iso6346_format"),
    )


class ContainerLocation(Base):
    """Slot history of a container; the current location has removed_at IS NULL."""
    __tablename__ = "container_location"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    container_no: Mapped[str] = mapped_column(ForeignKey("container.container_no"))
    slot_id: Mapped[int] = mapped_column(ForeignKey("slot.slot_id"))
    placed_at: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now())
    removed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP)
    __table_args__ = (
        CheckConstraint("removed_at IS NULL OR removed_at >= placed_at", name="removed_after_placed"),
        # One container per slot and one current location per container.
        Index("uq_location_current_slot", "slot_id", unique=True,
              postgresql_where=text("removed_at IS NULL")),
        Index("uq_location_current_container", "container_no", unique=True,
              postgresql_where=text("removed_at IS NULL")),
    )


# ─── Orders and crane moves ────────────────────────────────────────────────────

class Order(Base):
    """Inbound or outbound request at the gate."""
    __tablename__ = "orders"
    order_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    type: Mapped[OrderType] = mapped_column(_enum(OrderType))
    status: Mapped[OrderStatus] = mapped_column(_enum(OrderStatus),
                                                server_default=OrderStatus.QUEUED.value)
    # Set at creation for inbound; set when the container is picked for outbound.
    container_no: Mapped[str | None] = mapped_column(ForeignKey("container.container_no"))
    truck_no: Mapped[str | None] = mapped_column(String(20))
    # Policy that processed the order, e.g. "run_20261101-041443/best_model".
    decided_by: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP)
    __table_args__ = (
        CheckConstraint("type = 'outbound' OR container_no IS NOT NULL",
                        name="inbound_has_container"),
        CheckConstraint("status <> 'done' OR (container_no IS NOT NULL AND completed_at IS NOT NULL)",
                        name="done_is_complete"),
        Index("ix_orders_status", "status"),
        # Lookup of a container's arrival time.
        Index("ix_orders_container_completed", "container_no", "completed_at"),
    )


class MoveLog(Base):
    """Crane instructions of an order, in execution order."""
    __tablename__ = "move_log"
    move_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.order_id"))
    seq: Mapped[int] = mapped_column(SmallInteger)
    status: Mapped[MoveStatus] = mapped_column(_enum(MoveStatus),
                                               server_default=MoveStatus.PLANNED.value)
    move_type: Mapped[MoveType] = mapped_column(_enum(MoveType))
    container_no: Mapped[str] = mapped_column(ForeignKey("container.container_no"))
    from_slot_id: Mapped[int | None] = mapped_column(ForeignKey("slot.slot_id"))
    to_slot_id: Mapped[int | None] = mapped_column(ForeignKey("slot.slot_id"))
    crane_travel: Mapped[int] = mapped_column(SmallInteger, server_default=text("0"))
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now())
    done_at: Mapped[datetime | None] = mapped_column(TIMESTAMP)
    __table_args__ = (
        UniqueConstraint("order_id", "seq", name="uq_move_order_seq"),
        CheckConstraint(
            "(move_type = 'store'    AND from_slot_id IS NULL     AND to_slot_id IS NOT NULL) OR "
            "(move_type = 'relocate' AND from_slot_id IS NOT NULL AND to_slot_id IS NOT NULL) OR "
            "(move_type = 'retrieve' AND from_slot_id IS NOT NULL AND to_slot_id IS NULL)",
            name="slots_match_type"),
        CheckConstraint("status <> 'done' OR done_at IS NOT NULL", name="done_has_time"),
        CheckConstraint("crane_travel >= 0", name="travel_nonneg"),
    )
