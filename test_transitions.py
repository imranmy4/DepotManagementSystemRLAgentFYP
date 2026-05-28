import numpy as np
from depot_env import DepotEnv, DepotConfig


def fresh_env(seed=0, **cfg):
    env = DepotEnv(DepotConfig(**cfg))
    env.reset(seed=seed)
    return env


def test_inbound_places_at_lowest_free_tier():
    env = fresh_env()
    env.mode = 0
    env.n_in_remaining = 5  # ensure room to act

    env._apply_inbound(0, 0, 0)
    assert env.grid[0, 0, 0, 0] == 0, "First placement should be at tier 0"
    assert env.grid[0, 0, 0, 1] == -1, "Tier 1 should still be empty"

    env._apply_inbound(0, 0, 0)
    assert env.grid[0, 0, 0, 1] == 0, "Second placement should be at tier 1"
    print("Inbound places at lowest free tier ✓")


def test_inbound_decrements_queue():
    env = fresh_env()
    env.mode = 0
    env.n_in_remaining = 3
    env._apply_inbound(1, 1, 1)
    assert env.n_in_remaining == 2
    print("Inbound decrements n_in_remaining ✓")


def test_outbound_removes_container():
    env = fresh_env()
    env.mode = 1
    env.n_out_remaining = 5
    env.grid[0, 0, 0, 0] = 7

    env._apply_outbound(0, 0, 0, 0)
    assert env.grid[0, 0, 0, 0] == -1, "Slot should be empty after retrieval"
    assert env.n_out_remaining == 4
    print("Outbound removes container and decrements queue ✓")


def test_outbound_crane_distance():
    env = fresh_env()
    env.mode = 1
    env.n_out_remaining = 5
    env.crane_pos = np.array([0, 0, 0], dtype=np.int32)
    env.grid[2, 1, 3, 0] = 5

    dist, _ = env._apply_outbound(2, 1, 3, 0)
    assert dist == 2 + 1 + 3, f"Expected Manhattan distance 6, got {dist}"
    assert (env.crane_pos == [2, 1, 3]).all(), "Crane position should update"
    print("Outbound crane travel ✓")


def test_tick_advance_ages_containers():
    env = fresh_env()
    env.grid[0, 0, 0, 0] = 0
    env.grid[1, 1, 1, 0] = 3
    initial_tick = env.tick

    env._advance_tick()

    assert env.grid[0, 0, 0, 0] == 1, "Container should age by 1"
    assert env.grid[1, 1, 1, 0] == 4
    assert env.tick > initial_tick
    print("Tick advance ages occupied slots ✓")


def test_empty_slots_not_aged():
    env = fresh_env()
    env.grid[:] = -1
    env._advance_tick()
    assert (env.grid == -1).all(), "Empty slots should stay at -1"
    print("Tick advance leaves empty slots alone ✓")


def test_mode_switches_when_outbound_queue_empties():
    env = fresh_env()
    env.mode = 1
    env.n_out_remaining = 1
    env.n_in_remaining = 2
    env.grid[0, 0, 0, 0] = 1

    env._apply_outbound(0, 0, 0, 0)
    env._update_mode_or_advance()

    assert env.mode == 0, "Should switch to inbound when outbound queue empty"
    print("Mode switches outbound → inbound ✓")


def test_mode_switches_when_inbound_queue_empties():
    env = fresh_env()
    env.mode = 0
    env.n_in_remaining = 1
    env.n_out_remaining = 2
    env.grid[0, 0, 0, 0] = 1  # so yard has containers for outbound mode

    env._apply_inbound(1, 1, 1)
    env._update_mode_or_advance()

    assert env.mode == 1, "Should switch to outbound when inbound queue empty"
    print("Mode switches inbound → outbound ✓")


def test_tick_advances_when_both_queues_empty():
    env = fresh_env()
    env.mode = 0
    env.n_in_remaining = 1
    env.n_out_remaining = 0
    initial_tick = env.tick

    env._apply_inbound(0, 0, 0)
    env._update_mode_or_advance()

    assert env.tick > initial_tick, "Tick should advance when both queues empty"
    print("Tick advances when both queues empty ✓")


def test_full_episode_runs_to_termination():
    """Sanity check: random valid actions complete an episode without crashing."""
    env = fresh_env(n_ticks=10)
    done = False
    steps = 0

    while not done:
        mask = env.action_masks()
        if not mask.any():
            # No valid action — force tick advance (edge case, Step 3 doesn't handle full yard)
            env._advance_tick()
            done = env.tick >= env.config.n_ticks
            continue
        valid_actions = np.where(mask)[0]
        action = int(env.np_random.choice(valid_actions))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        assert steps < 10000, "Episode running too long, likely infinite loop"

    assert env.tick >= env.config.n_ticks
    print(f"Full episode ran to termination in {steps} steps ✓")


if __name__ == "__main__":
    test_inbound_places_at_lowest_free_tier()
    test_inbound_decrements_queue()
    test_outbound_removes_container()
    test_outbound_crane_distance()
    test_tick_advance_ages_containers()
    test_empty_slots_not_aged()
    test_mode_switches_when_outbound_queue_empties()
    test_mode_switches_when_inbound_queue_empties()
    test_tick_advances_when_both_queues_empty()
    test_full_episode_runs_to_termination()
    print("\nAll transition tests passed ✓")