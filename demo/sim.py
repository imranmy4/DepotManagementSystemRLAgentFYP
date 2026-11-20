"""Demo controls on top of the deployed depot system.

Everything an operator does goes through the product code: orders are created with
`operations` and queued in Redis with `order_queue`, the worker step (`worker.step`) plans them
with the deployed model, and moves are confirmed with `operations.confirm_move`. Only three
things are demo-specific and act on the database directly:

- build_yard:  wipe the yard and fill it to a chosen level, with container ages either
               "structured" (long-tailed, oldest at the bottom of each stack, as in training)
               or "uniform" (random age per container, unsorted: a messy yard);
- skip_time:   make time pass by moving every stored timestamp into the past;
- new_day:     random gate traffic for one day, U{0..max_in} inbound and U{0..max_out} outbound.
"""
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "api"))

from sqlalchemy import func, select, text  # noqa: E402

from app import order_queue, worker  # noqa: E402
from app.adapter import build_env, load_policy  # noqa: E402
from app.db import operations  # noqa: E402
from app.db.models import (  # noqa: E402
    MoveLog, MoveStatus, MoveType, Order, OrderStatus, OrderType,
)
from app.db.seed import model_source  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.iso6346 import check_digit  # noqa: E402
from app.settings import settings  # noqa: E402

SEED_LABEL = "demo: initial yard"
OWNERS = ["CSQU", "MSKU", "OOLU", "TGHU", "TCLU", "SEGU"]
TIME_COLUMNS = {
    "container": ["created_at"],
    "container_location": ["placed_at", "removed_at"],
    "orders": ["created_at", "completed_at"],
    "move_log": ["created_at", "done_at"],
}


@dataclass
class System:
    """The deployed model and a Redis client, loaded once per demo session."""
    policy: object
    decided_by: str
    redis: object

    @classmethod
    def connect(cls) -> "System":
        with SessionLocal() as s:
            policy = load_policy(s)
        return cls(policy, model_source(), order_queue.connect())


def container_no(rng: random.Random) -> str:
    """Random valid ISO 6346 freight container number."""
    first10 = rng.choice(OWNERS) + f"{rng.randrange(1_000_000):06d}"
    return first10 + str(check_digit(first10))


def _fresh_no(session, rng: random.Random) -> str:
    """A container number that is neither in the yard nor on an open inbound order."""
    in_use = set(operations.occupancy(session).values()) | set(session.scalars(
        select(Order.container_no).where(Order.type == OrderType.INBOUND,
                                         Order.status.in_([OrderStatus.QUEUED, OrderStatus.IN_PROGRESS]))))
    while True:
        no = container_no(rng)
        if no not in in_use:
            return no


# ─── Yard conditions ──────────────────────────────────────────────────────────

def initial_grid(shape, fill: float, mode: str, age_mean: float, age_max: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Yard as ages per slot (-1 empty), filled flat bottom-up to `fill` of capacity.

    structured: ages ~ exponential(age_mean) capped at age_max, sorted oldest at the bottom
    uniform:    ages ~ U{0..age_max} per container, in random order within a stack
    """
    B, Ba, R, H = shape
    n_stacks = B * Ba * R
    n_fill = int(round(min(max(fill, 0.0), 1.0) * n_stacks * H))
    base, extra = divmod(n_fill, n_stacks)
    heights = np.full(n_stacks, base)
    heights[rng.permutation(n_stacks)[:extra]] += 1
    occupied = np.arange(H) < heights.reshape(B, Ba, R)[..., None]
    if mode == "structured":
        ages = np.minimum(rng.exponential(max(age_mean, 1e-6), size=shape), age_max).astype(int)
    else:
        ages = rng.integers(0, age_max + 1, size=shape)
    grid = np.where(occupied, ages, -1)
    if mode == "structured":
        grid = -np.sort(-grid, axis=3)       # descending per stack: oldest at the bottom
    return grid


def clear_all(sys_: System) -> None:
    """Remove every order, move, container and location; crane back to its home stack."""
    with SessionLocal.begin() as s:
        s.execute(text("DELETE FROM move_log"))
        s.execute(text("DELETE FROM orders"))
        s.execute(text("DELETE FROM container_location"))
        s.execute(text("DELETE FROM container"))
        s.execute(text("UPDATE yard_state SET crane_block = 0, crane_bay = 0, crane_row = 0"))
    sys_.redis.delete(order_queue.OUTBOUND, order_queue.INBOUND, order_queue.PROCESSING)


def build_yard(sys_: System, fill: float, mode: str, age_mean: float, age_max: int,
               seed: int) -> int:
    """Start over with a pre-filled yard. Each container arrives through a normal inbound
    order (store move, confirmed), then its timestamps are moved back by its age.
    Returns the number of containers placed."""
    clear_all(sys_)
    cfg = sys_.policy.depot_config
    shape = (cfg.n_blocks, cfg.n_bays, cfg.n_rows, cfg.n_tiers)
    grid = initial_grid(shape, fill, mode, age_mean, age_max, np.random.default_rng(seed))
    rng = random.Random(seed)
    nos, secs = [], []
    with SessionLocal.begin() as s:
        used = set()
        for t in range(cfg.n_tiers):                       # bottom-up: never a floating box
            for b, ba, r in zip(*np.nonzero(grid[..., t] >= 0)):
                no = container_no(rng)
                while no in used:
                    no = container_no(rng)
                used.add(no)
                order = operations.create_inbound_order(s, no, owner_line=no[:3])
                operations.start_order(s, order.order_id, SEED_LABEL)
                pos = (int(b), int(ba), int(r), t)
                [move] = operations.plan_moves(
                    s, order.order_id, [operations.PlannedMove(MoveType.STORE, None, pos)])
                operations.confirm_move(s, move.move_id)
                nos.append(no)
                secs.append(float(grid[b, ba, r, t]) * settings.tick_seconds)
        if nos:
            for table, cols in TIME_COLUMNS.items():
                sets = ", ".join(f"{c} = {table}.{c} - make_interval(secs => v.s)" for c in cols)
                s.execute(text(
                    f"UPDATE {table} SET {sets} FROM unnest(CAST(:nos AS text[]), "
                    f"CAST(:secs AS float8[])) AS v(no, s) WHERE {table}.container_no = v.no"),
                    {"nos": nos, "secs": secs})
    return len(nos)


# ─── Time and traffic ─────────────────────────────────────────────────────────

def skip_time(days: float) -> None:
    """Let `days` model ticks pass: every stored timestamp moves that far into the past."""
    with SessionLocal.begin() as s:
        for table, cols in TIME_COLUMNS.items():
            sets = ", ".join(f"{c} = {c} - make_interval(secs => :s)" for c in cols)
            s.execute(text(f"UPDATE {table} SET {sets}"), {"s": days * settings.tick_seconds})


def submit_orders(sys_: System, n_in: int, n_out: int, rng: random.Random) -> int:
    """Create and queue orders exactly as the API does (commit first, then Redis)."""
    created = []
    with SessionLocal.begin() as s:
        for _ in range(n_out):
            created.append(operations.create_outbound_order(s, truck_no=f"T{rng.randrange(1000):03d}"))
        for _ in range(n_in):
            created.append(operations.create_inbound_order(
                s, _fresh_no(s, rng), truck_no=f"T{rng.randrange(1000):03d}"))
    for o in created:
        order_queue.submit(sys_.redis, o.order_id, o.type)
    return len(created)


def new_day(sys_: System, max_in: int, max_out: int, rng: random.Random) -> tuple[int, int]:
    """One day of random gate traffic, as in training: U{0..max_in} in, U{0..max_out} out."""
    n_in, n_out = rng.randint(0, max_in), rng.randint(0, max_out)
    submit_orders(sys_, n_in, n_out, rng)
    return n_in, n_out


def reject_queued() -> int:
    """Reject every queued order (their ids left in Redis are skipped by the worker)."""
    with SessionLocal.begin() as s:
        queued = operations.queued_orders(s)
        for o in queued:
            operations.reject_order(s, o.order_id, "cleared by demo operator")
    return len(queued)


def confirm_next() -> bool:
    """Confirm the next planned move of the order in progress; False if there is none."""
    with SessionLocal.begin() as s:
        order = operations.order_in_progress(s)
        if order is None:
            return False
        nxt = next((m for m in operations.order_moves(s, order.order_id)
                    if m.status == MoveStatus.PLANNED), None)
        if nxt is None:
            return False
        operations.confirm_move(s, nxt.move_id)
    return True


def plan_next(sys_: System) -> bool:
    """Worker steps until an order is planned. Each step takes one id from Redis; ids of
    orders rejected meanwhile are skipped, as are orders the worker rejects (yard full or
    empty). False when the queue is empty or the crane is busy."""
    while worker.step(sys_.redis, sys_.policy, sys_.decided_by):
        with SessionLocal() as s:
            if operations.order_in_progress(s) is not None:
                return True
    return False


def auto_step(sys_: System, auto_days: bool, max_in: int, max_out: int,
              rng: random.Random) -> str:
    """Advance the demo by one action: confirm a move, else plan an order, else (when
    auto_days) end the day and generate the next day's orders."""
    if confirm_next():
        return "move"
    if plan_next(sys_):
        return "plan"
    if auto_days:
        skip_time(1)
        new_day(sys_, max_in, max_out, rng)
        return "day"
    return "idle"


# ─── Reads for the page ───────────────────────────────────────────────────────

def snapshot(sys_: System):
    """Current yard as a DepotEnv (ages in ticks, crane), the order in progress with its
    moves as positions, recent orders and running totals."""
    with SessionLocal() as s:
        env = build_env(s, sys_.policy, OrderType.INBOUND)
        current = operations.order_in_progress(s)
        moves = []
        if current is not None:
            slots = operations.slot_positions(s)
            moves = [dict(move_id=m.move_id, type=m.move_type.value, status=m.status.value,
                          container_no=m.container_no, src=slots.get(m.from_slot_id),
                          dst=slots.get(m.to_slot_id), travel=m.crane_travel)
                     for m in operations.order_moves(s, current.order_id)]
            current = dict(order_id=current.order_id, type=current.type.value,
                           truck_no=current.truck_no, decided_by=current.decided_by)
        orders = [dict(order=o.order_id, type=o.type.value, status=o.status.value,
                       container=o.container_no or "", note=o.reject_reason or "")
                  for o in operations.recent_orders(s, 40) if o.decided_by != SEED_LABEL]
        demo_moves = select(MoveLog).join(Order, Order.order_id == MoveLog.order_id).where(
            MoveLog.status == MoveStatus.DONE, Order.decided_by != SEED_LABEL).subquery()
        stats = s.execute(select(
            func.count().filter(demo_moves.c.move_type == MoveType.RETRIEVE),
            func.count().filter(demo_moves.c.move_type == MoveType.RELOCATE),
            func.count().filter(demo_moves.c.move_type == MoveType.STORE),
            func.coalesce(func.sum(demo_moves.c.crane_travel), 0))).one()
        totals = dict(retrieved=stats[0], reshuffles=stats[1], stored=stats[2], travel=stats[3],
                      queued=len(operations.queued_orders(s)),
                      rejected=s.scalar(select(func.count()).select_from(Order).where(
                          Order.status == OrderStatus.REJECTED)))
    return env, current, moves, orders, totals
