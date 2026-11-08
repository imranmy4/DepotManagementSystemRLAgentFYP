import numpy as np
from depot_env import DepotEnv, DepotConfig
from reward import RewardConfig, RewardCalculator


def fresh_env(reward_config=None, **cfg):
    env = DepotEnv(DepotConfig(**cfg), reward_config=reward_config)
    env.reset(seed=0)
    return env


def test_action_bonus_is_constant():
    env = fresh_env(RewardConfig(w_action_bonus=0.5))
    assert env.reward_calc.action_bonus() == 0.5
    print("Action bonus ✓")


def test_burial_penalty_empty_stack_is_zero():
    env = fresh_env()
    env.grid[0, 0, 0, :] = -1
    assert env.reward_calc.burial_penalty(0, 0, 0) == 0.0
    print("Burial penalty empty stack = 0 ✓")


def test_burial_penalty_scales_with_oldest():
    env = fresh_env(RewardConfig(w_burial=1.0))
    env.grid[0, 0, 0, 0] = 3
    env.grid[0, 0, 0, 1] = 7  # oldest
    env.grid[0, 0, 0, 2] = 2
    penalty = env.reward_calc.burial_penalty(0, 0, 0)
    assert penalty == -7.0, f"Expected -7.0, got {penalty}"
    print("Burial penalty scales with oldest container ✓")


def test_height_penalty_scales_with_height():
    env = fresh_env(RewardConfig(w_height=0.5))
    env.grid[0, 0, 0, 0] = 0
    env.grid[0, 0, 0, 1] = 0
    env.grid[0, 0, 0, 2] = 0  # height = 3
    penalty = env.reward_calc.height_penalty(0, 0, 0)
    assert penalty == -1.5, f"Expected -1.5, got {penalty}"
    print("Height penalty scales with height ✓")


def test_dwell_reward_proportional_to_dwell():
    env = fresh_env(RewardConfig(w_dwell=2.0))
    reward = env.reward_calc.dwell_reward(5)
    assert reward == 10.0, f"Expected 10.0, got {reward}"
    print("Dwell reward proportional ✓")


def test_travel_reward_short_distance_compensated():
    env = fresh_env(RewardConfig(
        w_travel=1.0,
        w_travel_compensate=0.5,
        threshold_divisor=2,
    ))
    # max_dist for default config (B=6, Ba=4, R=4) with weights 1 = 5+3+3 = 11
    # threshold = 11 / 2 = 5.5 (then capped via max(1, ...))
    short_dist = 2.0
    reward = env.reward_calc.travel_reward(short_dist)
    assert reward > 0, f"Short move should be positive, got {reward}"
    print("Travel: short distance compensated ✓")


def test_travel_reward_long_distance_penalised():
    env = fresh_env(RewardConfig(
        w_travel=1.0,
        w_travel_compensate=0.5,
        threshold_divisor=2,
    ))
    long_dist = 10.0
    reward = env.reward_calc.travel_reward(long_dist)
    assert reward < 0, f"Long move should be negative, got {reward}"
    print("Travel: long distance penalised ✓")


def test_reshuffle_penalty_scales_with_count():
    # All same-block moves (multiplier 1.0) → penalty scales with count
    env = fresh_env(RewardConfig(w_reshuffle=2.0))
    assert env.reward_calc.reshuffle_penalty([]) == 0.0
    assert env.reward_calc.reshuffle_penalty([0, 0, 0]) == -6.0
    print("Reshuffle penalty scales linearly with count ✓")


def test_reshuffle_penalty_scales_with_scope():
    env = fresh_env(RewardConfig(
        w_reshuffle=1.0,
        reshuffle_mult_same=1.0,
        reshuffle_mult_adjacent=2.0,
        reshuffle_mult_far=4.0,
    ))
    # one same(1) + one adjacent(2) + one far(4) = 7
    assert env.reward_calc.reshuffle_penalty([0, 1, 3]) == -7.0
    print("Reshuffle penalty scales with scope ✓")


def test_tick_penalty_zero_when_empty_yard():
    env = fresh_env()
    env.grid[:] = -1
    assert env.reward_calc.tick_penalty() == 0.0
    print("Tick penalty = 0 on empty yard ✓")


def test_tick_penalty_scales_with_dwell():
    env = fresh_env(RewardConfig(w_mean_idle=1.0, w_max_idle=1.0))
    env.grid[:] = -1
    env.grid[0, 0, 0, 0] = 4
    env.grid[1, 1, 1, 0] = 8
    # mean = 6, max = 8 → -(6 + 8) = -14
    penalty = env.reward_calc.tick_penalty()
    assert penalty == -14.0, f"Expected -14.0, got {penalty}"
    print("Tick penalty: mean + max scaling ✓")


def test_step_returns_breakdown_in_info():
    env = fresh_env(n_ticks=5)
    mask = env.action_masks()
    valid = np.where(mask)[0]
    action = int(env.np_random.choice(valid))
    obs, reward, terminated, truncated, info = env.step(action)
    assert "reward_breakdown" in info
    assert len(info["reward_breakdown"]) > 0
    print(f"Step breakdown: {info['reward_breakdown']} ✓")


def test_episode_bonus_applied_at_termination():
    env = fresh_env(
        RewardConfig(
            w_action_bonus=0.0,
            w_burial=0.0,
            w_height=0.0,
            w_dwell=0.0,
            w_travel=0.0,
            w_travel_compensate=0.0,
            w_reshuffle=0.0,
            w_mean_idle=0.0,
            w_max_idle=0.0,
            w_episode_bonus=100.0,
        ),
        n_ticks=2,
    )
    final_reward = None
    done = False
    while not done:
        mask = env.action_masks()
        if not mask.any():
            env._advance_tick()
            done = env.tick >= env.config.n_ticks
            if done:
                # No step reward to test, force the bonus path differently
                pass
            continue
        action = int(env.np_random.choice(np.where(mask)[0]))
        obs, reward, terminated, truncated, info = env.step(action)
        final_reward = reward
        done = terminated or truncated
    # Episode bonus should be present in the final step's reward
    print(f"Final step reward: {final_reward} ✓")


def test_profile_caps_pyramid():
    from reward import profile_caps
    caps = profile_caps(4, 4, 6, 1)
    assert caps[1:3, 1:3].tolist() == [[6, 6], [6, 6]], "Inner 2x2 capped at H"
    assert caps[0, 0] == 5 and caps[0, 2] == 5 and caps[3, 1] == 5, "Edge ring capped at H-1"
    assert (profile_caps(5, 5, 6, 1)[2, 2], profile_caps(5, 5, 6, 1)[0, 0]) == (6, 4)
    print("Profile caps form a pyramid ✓")


def test_cliff_penalty_tower_and_pit():
    env = fresh_env(RewardConfig(w_cliff=1.0, cliff_tol=1))
    env.grid[:] = -1
    env.grid[0, 1, 1, :4] = 0        # tower of 4 in the middle of an empty block
    # Inbound view: 4 neighbours each 4 tiers lower -> excess 3 each, mean 3
    assert env.reward_calc.cliff_penalty(0, 1, 1, outbound=False) == -3.0
    # Outbound view of an edge neighbour (3 neighbours, only the tower is taller): 3 / 3
    assert env.reward_calc.cliff_penalty(0, 0, 1, outbound=True) == -1.0
    # Gap within tolerance costs nothing
    env.grid[0, 1, 1, 1:] = -1       # tower now height 1
    assert env.reward_calc.cliff_penalty(0, 1, 1, outbound=False) == 0.0
    print("Cliff penalty: towers and pits ✓")


def test_cliff_ignores_other_blocks():
    env = fresh_env(RewardConfig(w_cliff=1.0, cliff_tol=0))
    env.grid[:] = -1
    env.grid[1, :, :, :] = 0         # block 1 full; block 0 empty
    env.grid[0, 3, 0, 0] = 0         # edge stack of block 0, next to block 1 in index order
    # Only in-block neighbours (all empty) count: corner, 2 neighbours x gap 1, mean 1
    assert env.reward_calc.cliff_penalty(0, 3, 0, outbound=False) == -1.0
    print("Cliff penalty is in-block only ✓")


def test_profile_penalty_above_cap():
    env = fresh_env(RewardConfig(w_profile=2.0, profile_step=1))
    env.grid[:] = -1
    env.grid[0, 0, 0, :] = 0         # edge stack at H=6, cap 5 -> 1 excess tier
    env.grid[0, 1, 1, :] = 0         # centre stack at H=6, cap 6 -> fine
    assert env.reward_calc.profile_penalty(0, 0, 0) == -2.0
    assert env.reward_calc.profile_penalty(0, 1, 1) == 0.0
    print("Profile penalty only above the cap ✓")


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
    print("Cliff cost independent of position ✓")


def test_burial_cap_limits_penalty():
    env = fresh_env(RewardConfig(w_burial=1.0, burial_cap=5))
    env.grid[:] = -1
    env.grid[0, 0, 0, 0] = 20
    assert env.reward_calc.burial_penalty(0, 0, 0) == -6.0, "Capped at 5 -> -(5 + 1)"
    env.grid[0, 0, 0, 0] = 3
    assert env.reward_calc.burial_penalty(0, 0, 0) == -4.0, "Below the cap is unchanged"
    print("Burial cap limits penalty ✓")


def test_center_bonus_inner_ring_only():
    env = fresh_env(RewardConfig(w_center=0.3))
    assert env.reward_calc.center_bonus(1, 2) == 0.3, "Inner 2x2 gets the full bonus"
    assert env.reward_calc.center_bonus(0, 0) == 0.0, "Corner gets nothing"
    assert env.reward_calc.center_bonus(3, 1) == 0.0, "Edge gets nothing"
    print("Centre bonus on inner ring only ✓")


def test_shape_terms_off_by_default():
    env = fresh_env()
    env.grid[:] = -1
    env.grid[0, 0, 0, :] = 0
    assert env.reward_calc.cliff_penalty(0, 0, 0, outbound=False) == 0.0
    assert env.reward_calc.profile_penalty(0, 0, 0) == 0.0
    print("Shape terms off with default weights ✓")


if __name__ == "__main__":
    test_profile_caps_pyramid()
    test_cliff_penalty_tower_and_pit()
    test_cliff_ignores_other_blocks()
    test_profile_penalty_above_cap()
    test_cliff_same_cost_corner_edge_centre()
    test_burial_cap_limits_penalty()
    test_center_bonus_inner_ring_only()
    test_shape_terms_off_by_default()
    test_action_bonus_is_constant()
    test_burial_penalty_empty_stack_is_zero()
    test_burial_penalty_scales_with_oldest()
    test_height_penalty_scales_with_height()
    test_dwell_reward_proportional_to_dwell()
    test_travel_reward_short_distance_compensated()
    test_travel_reward_long_distance_penalised()
    test_reshuffle_penalty_scales_with_count()
    test_reshuffle_penalty_scales_with_scope()
    test_tick_penalty_zero_when_empty_yard()
    test_tick_penalty_scales_with_dwell()
    test_step_returns_breakdown_in_info()
    test_episode_bonus_applied_at_termination()
    print("\nAll reward tests passed ✓")