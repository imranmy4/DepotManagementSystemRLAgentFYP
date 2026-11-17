"""Tests for the DB ⇄ env adapter: yard rebuild, action -> moves, and full order runs with the
deployed model."""
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from depot import paths
from depot.config import load_run_config

from app.adapter import NoValidAction, build_env, decide, dwell_ticks, load_policy
from app.db.models import MoveType, OrderType
from app.db.operations import (
    confirm_move, create_inbound_order, create_outbound_order, occupancy, plan_moves,
    start_order,
)
from app.db.seed import seed_layout
from helpers import StubPolicy, container_no, pass_days

T0 = datetime(2026, 11, 1, 8, 0, tzinfo=timezone.utc)
DAY = 86400


def run_order(session, policy, order):
    """Decide, plan and confirm every move of `order`; returns the planned moves."""
    start_order(session, order.order_id, "test")
    moves = decide(session, policy, order)
    for row in plan_moves(session, order.order_id, moves):
        confirm_move(session, row.move_id)
    return moves


def test_dwell_ticks():
    assert dwell_ticks(T0, T0 + timedelta(days=2, hours=23), DAY) == 2
    assert dwell_ticks(T0, T0 - timedelta(hours=1), DAY) == 0
    assert dwell_ticks(T0, T0 + timedelta(minutes=5), 60) == 5


def test_build_env_mirrors_yard(yard):
    run_order(yard, StubPolicy(), create_inbound_order(yard, "AAAU0000007"))
    pass_days(yard, 1)
    run_order(yard, StubPolicy(), create_inbound_order(yard, "BBBU0000000"))
    pass_days(yard, 2.5)

    env = build_env(yard, StubPolicy(), OrderType.OUTBOUND)
    assert env.grid[0, 0, 0, 0] == 3 and env.grid[0, 0, 0, 1] == 2
    assert (env.grid >= 0).sum() == 2
    assert env.mode == 1 and env.n_out_remaining == 1
    assert env.action_masks()[env.n_inbound:].any()


def test_inbound_store_goes_to_next_free_tier(yard):
    policy = StubPolicy(action=0)              # inbound action 0 = stack (0, 0, 0)
    run_order(yard, policy, create_inbound_order(yard, "AAAU0000007"))
    moves = run_order(yard, policy, create_inbound_order(yard, "BBBU0000000"))
    assert moves[0].move_type == MoveType.STORE and moves[0].to_pos == (0, 0, 0, 1)


def test_outbound_dig_matches_env_transition(yard):
    for cno in ["AAAU0000007", "BBBU0000000", "CCCU0000006"]:
        run_order(yard, StubPolicy(action=0), create_inbound_order(yard, cno))
        pass_days(yard, 1)

    env = build_env(yard, StubPolicy(), OrderType.OUTBOUND)
    bottom = env.n_inbound + 0                 # outbound action for slot (0, 0, 0, 0)
    env._apply_outbound(0, 0, 0, 0)            # what the env itself would do

    moves = run_order(yard, StubPolicy(action=bottom), create_outbound_order(yard))
    assert [m.move_type for m in moves] == [MoveType.RELOCATE] * 2 + [MoveType.RETRIEVE]
    after = build_env(yard, StubPolicy(), OrderType.INBOUND)
    np.testing.assert_array_equal(after.grid >= 0, env.grid >= 0)
    assert "AAAU0000007" not in occupancy(yard).values()


def test_outbound_from_empty_yard_raises(yard):
    order = create_outbound_order(yard)
    with pytest.raises(NoValidAction, match="no container"):
        decide(yard, StubPolicy(), order)


@pytest.mark.skipif(not (paths.DEPLOY_MODEL_DIR / "config_used.ini").exists(),
                    reason="no promoted model in deploy/model")
def test_deployed_model_runs_orders(session):
    """Real model on its real layout: a month of mixed orders, every plan valid."""
    seed_layout(session, load_run_config(paths.DEPLOY_MODEL_DIR)[0])
    policy = load_policy(session)
    rng = random.Random(0)
    for _ in range(30):
        for _ in range(rng.randint(0, 6)):
            if occupancy(session):
                run_order(session, policy, create_outbound_order(session))
        for _ in range(rng.randint(0, 8)):
            run_order(session, policy, create_inbound_order(session, container_no(rng)))
        pass_days(session, 1)
    assert len(occupancy(session)) > 0
