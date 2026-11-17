"""Tests for the reward terms and how the env applies them."""
import configparser

import numpy as np

from depot.config import load_dataclass
from depot.env import DepotEnv, DepotConfig
from depot.reward import RewardConfig


def fresh_env(reward_config=None, **cfg):
    env = DepotEnv(DepotConfig(**cfg), reward_config=reward_config)
    env.reset(seed=0)
    return env


def test_burial_free_on_empty_stack():
    env = fresh_env(RewardConfig(w_burial=1.0))
    env.grid[0, 0, 0, :] = -1
    env.grid[0, 0, 0, 0] = 0                 # the container just placed
    assert env.reward_calc.burial_penalty(0, 0, 0) == 0.0


def test_burial_scales_with_oldest_below():
    env = fresh_env(RewardConfig(w_burial=1.0, burial_cap=100))
    env.grid[0, 0, 0, :] = -1
    env.grid[0, 0, 0, :4] = [3, 7, 2, 0]     # new container (0) on top; oldest below = 7
    assert env.reward_calc.burial_penalty(0, 0, 0) == -8.0


def test_burial_on_fresh_stack_still_costs():
    env = fresh_env(RewardConfig(w_burial=2.0))
    env.grid[0, 0, 0, :] = -1
    env.grid[0, 0, 0, :2] = [0, 0]
    assert env.reward_calc.burial_penalty(0, 0, 0) == -2.0


def test_burial_cap_limits_penalty():
    env = fresh_env(RewardConfig(w_burial=1.0, burial_cap=5))
    env.grid[0, 0, 0, :] = -1
    env.grid[0, 0, 0, :2] = [20, 0]
    assert env.reward_calc.burial_penalty(0, 0, 0) == -6.0, "Capped at 5 -> -(5 + 1)"
    env.grid[0, 0, 0, :2] = [3, 0]
    assert env.reward_calc.burial_penalty(0, 0, 0) == -4.0, "Below the cap is unchanged"


def test_inbound_step_on_empty_stack_has_no_burial():
    env = fresh_env(RewardConfig(w_burial=2.0, w_cliff=0.0))
    env.grid[:] = -1
    env.mode, env.n_in_remaining, env.n_out_remaining = 0, 2, 0
    _, _, _, _, info = env.step(0)            # inbound into empty stack (0, 0, 0)
    assert info["reward_breakdown"]["burial"] == 0.0


def test_dwell_reward_is_log_of_age():
    env = fresh_env(RewardConfig(w_dwell=2.0))
    assert env.reward_calc.dwell_reward(0) == 0.0
    assert np.isclose(env.reward_calc.dwell_reward(5), 2.0 * np.log(6))


def test_default_weights_follow_operator_ranking():
    """Full travel range < one reshuffle; full age range (0-365 days) < full travel range."""
    env = fresh_env()
    c, max_dist = env.reward_calc.config, 3 + 3 + 3       # 2x3 block grid, 4x4 bays/rows
    assert c.w_travel * max_dist < c.w_reshuffle * c.reshuffle_mult_same
    assert env.reward_calc.dwell_reward(365) < c.w_travel * max_dist


def test_travel_penalty_linear():
    env = fresh_env(RewardConfig(w_travel=0.5))
    assert env.reward_calc.travel_penalty(0) == 0.0
    assert env.reward_calc.travel_penalty(4) == -2.0


def test_reshuffle_penalty_scales_with_count():
    env = fresh_env(RewardConfig(w_reshuffle=2.0))
    assert env.reward_calc.reshuffle_penalty([]) == 0.0
    assert env.reward_calc.reshuffle_penalty([0, 0, 0]) == -6.0


def test_reshuffle_penalty_scales_with_scope():
    env = fresh_env(RewardConfig(w_reshuffle=1.0, reshuffle_mult_same=1.0,
                                 reshuffle_mult_adjacent=2.0, reshuffle_mult_far=4.0))
    assert env.reward_calc.reshuffle_penalty([0, 1, 3]) == -7.0


def test_tick_penalty_zero_when_empty_yard():
    env = fresh_env()
    env.grid[:] = -1
    assert env.reward_calc.tick_penalty() == 0.0


def test_tick_penalty_is_mean_log_age():
    env = fresh_env(RewardConfig(w_mean_idle=1.0))
    env.grid[:] = -1
    env.grid[0, 0, 0, 0] = 4
    env.grid[1, 1, 1, 0] = 8
    assert np.isclose(env.reward_calc.tick_penalty(), -(np.log(5) + np.log(9)) / 2)


def test_every_skipped_tick_is_penalised():
    """A step that ends several ticks pays the idle penalty of each, as containers age."""
    env = fresh_env(RewardConfig(w_mean_idle=1.0), n_ticks=30)
    env.grid[:] = -1
    env.grid[0, 0, 0, 0] = 10
    env._tick_penalty = 0.0
    env._end_tick()
    env._end_tick()
    assert np.isclose(env._tick_penalty, -(np.log(12) + np.log(13)))
    assert env.grid[0, 0, 0, 0] == 12


def test_reset_skipped_ticks_age_containers():
    env = DepotEnv(DepotConfig(warmstart=0.5, max_in=0, max_out=0, n_ticks=5))
    env.reset(seed=0)                        # no demand: every tick is skipped
    assert env.tick == 5
    assert env.grid[env.grid >= 0].min() == 5
    assert env._tick_penalty == 0.0


def test_step_returns_breakdown_in_info():
    env = fresh_env(n_ticks=5)
    action = int(env.np_random.choice(np.flatnonzero(env.action_masks())))
    _, _, _, _, info = env.step(action)
    assert len(info["reward_breakdown"]) > 0


def test_cliff_penalty_tower_and_pit():
    env = fresh_env(RewardConfig(w_cliff=1.0, cliff_tol=1))
    env.grid[:] = -1
    env.grid[0, 1, 1, :4] = 0        # tower of 4 in the middle of an empty block
    # Inbound view: 4 neighbours each 4 tiers lower -> excess 3 each, mean 3
    assert env.reward_calc.cliff_penalty(0, 1, 1, outbound=False) == -3.0
    # Outbound view of an edge neighbour (3 neighbours, only the tower is taller): 3 / 3
    assert env.reward_calc.cliff_penalty(0, 0, 1, outbound=True) == -1.0
    env.grid[0, 1, 1, 1:] = -1       # tower now height 1: within tolerance
    assert env.reward_calc.cliff_penalty(0, 1, 1, outbound=False) == 0.0


def test_cliff_ignores_other_blocks():
    env = fresh_env(RewardConfig(w_cliff=1.0, cliff_tol=0))
    env.grid[:] = -1
    env.grid[1, :, :, :] = 0         # block 1 full; block 0 empty
    env.grid[0, 3, 0, 0] = 0         # edge stack of block 0, next to block 1 in index order
    # Only in-block neighbours (all empty) count: corner, 2 neighbours x gap 1, mean 1
    assert env.reward_calc.cliff_penalty(0, 3, 0, outbound=False) == -1.0


def test_cliff_same_cost_corner_edge_centre():
    """A stack 2 tiers above all its neighbours pays the same wherever it is."""
    env = fresh_env(RewardConfig(w_cliff=15.0, cliff_tol=1))
    costs = []
    for s in [(0, 0), (0, 1), (1, 1)]:              # corner, edge, centre
        env.grid[:] = -1
        env.grid[0, :, :, :3] = 0                   # flat block at height 3
        env.grid[0, s[0], s[1], :5] = 0             # this stack at height 5
        costs.append(env.reward_calc.cliff_penalty(0, *s, outbound=False))
    assert costs == [-15.0, -15.0, -15.0], costs


def test_old_snapshot_with_removed_fields_loads():
    parser = configparser.ConfigParser()
    parser["reward"] = {"w_burial": "2.0", "w_height": "0.0", "threshold_divisor": "100",
                        "w_action_bonus": "0.0", "w_max_idle": "0.0"}
    cfg = load_dataclass(RewardConfig, parser, "reward")
    assert cfg.w_burial == 2.0
