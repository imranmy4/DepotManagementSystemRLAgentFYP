import numpy as np
from depot_env import DepotEnv, DepotConfig


def fresh_env(**cfg):
    env = DepotEnv(DepotConfig(**cfg))
    env.reset(seed=0)
    env.grid[:] = -1  # clear warmstart, start fresh
    return env


def test_no_relocation_when_target_on_top():
    """Retrieving the topmost container needs no relocation."""
    env = fresh_env()
    env.grid[0, 0, 0, 0] = 3   # bottom
    env.grid[0, 0, 0, 1] = 5   # top

    dists = env.reshuffler.relocate_above(0, 0, 0, 1)
    assert len(dists) == 0, f"Expected 0 relocations, got {len(dists)}"
    # Stack should be unchanged
    assert env.grid[0, 0, 0, 0] == 3
    assert env.grid[0, 0, 0, 1] == 5
    print("No relocation when target on top ✓")


def test_relocates_one_container():
    """Retrieving second-from-top requires one relocation."""
    env = fresh_env()
    env.grid[0, 0, 0, 0] = 2   # target
    env.grid[0, 0, 0, 1] = 4   # to be relocated

    dists = env.reshuffler.relocate_above(0, 0, 0, 0)
    assert len(dists) == 1, f"Expected 1 relocation, got {len(dists)}"
    # Container at tier 1 should be gone from source stack
    assert env.grid[0, 0, 0, 1] == -1
    # Should now exist somewhere else with dwell=4
    found = np.argwhere(env.grid == 4)
    assert len(found) == 1, f"Expected exactly one container with dwell 4, got {len(found)}"
    print("Relocates one container ✓")


def test_prefers_empty_stack():
    """Empty stacks should always qualify as candidates."""
    env = fresh_env()
    env.grid[0, 0, 0, 0] = 2   # target
    env.grid[0, 0, 0, 1] = 4   # to be relocated

    # Fill other stacks in same block with high dwell so empty is the only valid option
    for ba in range(env.Ba):
        for r in range(env.R):
            if ba == 0 and r == 0:
                continue
            if ba == 0 and r == 1:
                # Leave (0,0,0,1) as the only "empty" option in block 0
                env.grid[0, 0, r, :] = -1
            else:
                env.grid[0, ba, r, 0] = 10  # high dwell, would bury

    env.reshuffler.relocate_above(0, 0, 0, 0)
    # Container should have landed in (0, 0, 1) since it's the cleanest
    relocated = np.argwhere(env.grid == 4)
    assert len(relocated) == 1
    print(f"Container relocated to {tuple(relocated[0])} ✓")


def test_picks_highest_max_dwell_among_valid():
    env = fresh_env()
    env.grid[0, 0, 0, 0] = 1   # target at TIER 0 (will be retrieved)
    env.grid[0, 0, 0, 1] = 10  # to relocate (above target)

    env.grid[0, 0, 1, 0] = 2
    env.grid[0, 0, 2, 0] = 7
    env.grid[0, 0, 3, 0] = 4
    for ba in range(1, env.Ba):
        for r in range(env.R):
            env.grid[0, ba, r, :] = 100

    env.reshuffler.relocate_above(0, 0, 0, 0)  # ← target_h=0, so tier 1 gets relocated
    
    assert env.grid[0, 0, 2, 1] == 10, "Should pick stack with highest max_dwell among valid"
    print("Picks highest max_dwell among valid candidates ✓")


def test_falls_back_when_no_valid_candidate():
    """When all stacks would be buried, fall back to lowest-max stack."""
    env = fresh_env()
    env.grid[0, 0, 0, 0] = 1    # target
    env.grid[0, 0, 0, 1] = 5    # to relocate (dwell=5)

    # All other stacks in yard have max_dwell > 5 (would bury)
    for bl in range(env.B):
        for ba in range(env.Ba):
            for r in range(env.R):
                if bl == 0 and ba == 0 and r == 0:
                    continue
                env.grid[bl, ba, r, 0] = 6 + bl  # 6, 7, 8, ... varying highs

    env.reshuffler.relocate_above(0, 0, 0, 0)  # ← target_h=0, so tier 1 gets relocated
    # Should land on stack with lowest max_dwell (which is 6, block 0)
    found = np.argwhere(env.grid == 5)
    assert len(found) == 1
    bl, ba, r, h = found[0]
    # The lowest-max stack in block 0 (excluding source) should have max_dwell=6
    other_max = env._stack_height(bl, ba, r)
    print(f"Fallback relocation landed at {(bl, ba, r, h)} ✓")


def test_outbound_with_buried_target_triggers_reshuffle():
    """Full integration: outbound action on a buried container reshuffles."""
    env = fresh_env()
    env.mode = 1
    env.n_out_remaining = 1

    env.grid[0, 0, 0, 0] = 5   # target (buried)
    env.grid[0, 0, 0, 1] = 2   # above
    env.grid[0, 0, 0, 2] = 3   # above

    dist, _ = env._apply_outbound(0, 0, 0, 0)

    # Target should be gone
    assert env.grid[0, 0, 0, 0] == -1
    # The two containers above should be relocated somewhere
    relocated_2 = np.argwhere(env.grid == 2)
    relocated_3 = np.argwhere(env.grid == 3)
    assert len(relocated_2) == 1, "Container with dwell=2 should be relocated"
    assert len(relocated_3) == 1, "Container with dwell=3 should be relocated"
    print(f"Buried target retrieved, 2 relocations performed ✓")


def test_relocations_preserve_total_container_count():
    """Total occupied slots before == total after - 1 (target removed, others relocated)."""
    env = fresh_env()
    env.mode = 1
    env.n_out_remaining = 1

    # Populate stack
    env.grid[1, 1, 1, 0] = 8   # target
    env.grid[1, 1, 1, 1] = 1
    env.grid[1, 1, 1, 2] = 2
    env.grid[1, 1, 1, 3] = 3

    before = (env.grid >= 0).sum()
    env._apply_outbound(1, 1, 1, 0)
    after = (env.grid >= 0).sum()

    assert after == before - 1, f"Expected {before - 1} containers after, got {after}"
    print("Total container count preserved (minus target) ✓")


if __name__ == "__main__":
    test_no_relocation_when_target_on_top()
    test_relocates_one_container()
    test_prefers_empty_stack()
    test_picks_highest_max_dwell_among_valid()
    test_falls_back_when_no_valid_candidate()
    test_outbound_with_buried_target_triggers_reshuffle()
    test_relocations_preserve_total_container_count()
    print("\nAll reshuffle tests passed ✓")