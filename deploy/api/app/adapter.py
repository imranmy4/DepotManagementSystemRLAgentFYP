"""DB ⇄ env adapter: rebuilds the live yard from Postgres as a DepotEnv, lets the policy decide
an order, and turns the decision into crane moves for operations.plan_moves.

    policy = load_policy(session)                  # deployed model + layout check
    moves = decide(session, policy, order)         # -> list[PlannedMove]
    plan_moves(session, order.order_id, moves)
"""
from datetime import datetime

import numpy as np
from sqlalchemy.orm import Session

from depot import paths
from depot.env import DepotEnv
from depot.inference import Policy

from app.db.models import MoveType, Order, OrderType
from app.db.operations import PlannedMove, crane_position, current_yard, db_now
from app.db.seed import LayoutMismatch, layout_diff
from app.settings import settings

# Tick fed to the policy: the observation still contains it, and the middle of a training
# episode avoids end-of-episode behaviour (until tick is removed from the observation).
DEPLOY_TICK = 15


class NoValidAction(RuntimeError):
    """The order cannot be served in the current yard (outbound from empty, inbound into full)."""


def load_policy(session: Session) -> Policy:
    """Load the deployed model and check that the database layout matches its yard."""
    policy = Policy.load(paths.DEPLOY_MODEL_DIR)
    diff = layout_diff(session, policy.depot_config)
    if diff:
        raise LayoutMismatch(f"database layout does not match the deployed model: {diff}")
    return policy


def dwell_ticks(arrived_at: datetime, now: datetime, tick_seconds: int) -> int:
    """Whole ticks since yard arrival (never negative)."""
    return max(0, int((now - arrived_at).total_seconds() // tick_seconds))


def build_env(session: Session, policy: Policy, order_type: OrderType) -> DepotEnv:
    """DepotEnv holding the current yard, crane and order mode, ready for policy.act."""
    now = db_now(session)
    cfg = policy.depot_config
    env = DepotEnv(cfg, reward_config=policy.reward_config)

    env.grid = np.full((cfg.n_blocks, cfg.n_bays, cfg.n_rows, cfg.n_tiers), -1, dtype=np.int32)
    for c in current_yard(session):
        env.grid[c.pos] = dwell_ticks(c.arrived_at, now, settings.tick_seconds)
    env.crane_pos = np.array(crane_position(session), dtype=np.int32)
    env.tick = DEPLOY_TICK
    inbound = order_type == OrderType.INBOUND
    env.mode = 0 if inbound else 1
    env.n_in_remaining, env.n_out_remaining = (1, 0) if inbound else (0, 1)
    return env


def action_to_moves(env: DepotEnv, action: int) -> list[PlannedMove]:
    """Crane moves for an env action: one store, or relocations followed by one retrieve."""
    kind, *pos = env.decode_action(action)
    pos = tuple(int(x) for x in pos)
    if kind == "inbound":
        bl, ba, r = pos
        return [PlannedMove(MoveType.STORE, None, (bl, ba, r, int(env._stack_height(bl, ba, r))))]

    bl, ba, r, h = pos
    # Plain ints: the reshuffler returns numpy integers, which the DB driver cannot bind.
    moves = [PlannedMove(MoveType.RELOCATE, tuple(map(int, m[1:5])), tuple(map(int, m[5:9])))
             for m in env.plan_relocations(bl, ba, r, h)]
    travel = env.crane_dist(env.crane_pos, (bl, ba, r))
    moves.append(PlannedMove(MoveType.RETRIEVE, (bl, ba, r, h), None, crane_travel=travel))
    return moves


def decide(session: Session, policy: Policy, order: Order) -> list[PlannedMove]:
    """Let the policy choose where to store / which container to release for `order`."""
    env = build_env(session, policy, order.type)
    if not env.action_masks().any():
        raise NoValidAction("yard is full" if order.type == OrderType.INBOUND
                            else "no container in the yard")
    return action_to_moves(env, policy.act(env))
