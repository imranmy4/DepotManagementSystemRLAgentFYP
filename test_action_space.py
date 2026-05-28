import numpy as np
from depot_env import DepotEnv, DepotConfig


def test_encode_decode_inbound_roundtrip():
    """Every inbound encoding should decode back to the original tuple."""
    env = DepotEnv(DepotConfig())
    for bl in range(env.B):
        for ba in range(env.Ba):
            for r in range(env.R):
                idx = env.encode_inbound(bl, ba, r)
                kind, bl2, ba2, r2 = env.decode_action(idx)
                assert kind == "inbound"
                assert (bl, ba, r) == (bl2, ba2, r2), f"Mismatch at ({bl},{ba},{r})"
    print("Inbound encode/decode roundtrip ✓")


def test_encode_decode_outbound_roundtrip():
    """Every outbound encoding should decode back to the original tuple."""
    env = DepotEnv(DepotConfig())
    for bl in range(env.B):
        for ba in range(env.Ba):
            for r in range(env.R):
                for h in range(env.H):
                    idx = env.encode_outbound(bl, ba, r, h)
                    kind, bl2, ba2, r2, h2 = env.decode_action(idx)
                    assert kind == "outbound"
                    assert (bl, ba, r, h) == (bl2, ba2, r2, h2), \
                        f"Mismatch at ({bl},{ba},{r},{h})"
    print("Outbound encode/decode roundtrip ✓")


def test_action_indices_partition():
    """
    Inbound indices fill [0, n_inbound), outbound fills [n_inbound, n_actions).
    No gaps, no overlaps.
    """
    env = DepotEnv(DepotConfig())

    inbound_indices = set()
    for bl in range(env.B):
        for ba in range(env.Ba):
            for r in range(env.R):
                inbound_indices.add(env.encode_inbound(bl, ba, r))
    assert inbound_indices == set(range(env.n_inbound)), \
        "Inbound indices don't fill [0, n_inbound)"

    outbound_indices = set()
    for bl in range(env.B):
        for ba in range(env.Ba):
            for r in range(env.R):
                for h in range(env.H):
                    outbound_indices.add(env.encode_outbound(bl, ba, r, h))
    assert outbound_indices == set(range(env.n_inbound, env.n_actions)), \
        "Outbound indices don't fill [n_inbound, n_actions)"

    assert inbound_indices & outbound_indices == set(), "Overlap detected"
    print("Action index partition ✓")


def test_mask_inbound_empty_yard():
    """In inbound mode with empty yard, all inbound actions are valid."""
    env = DepotEnv(DepotConfig())
    env.reset(seed=0)
    env.mode = 0  # force inbound

    mask = env.action_masks()
    assert mask[:env.n_inbound].all(), "All inbound slots should be valid"
    assert not mask[env.n_inbound:].any(), "All outbound slots should be masked"
    print("Mask: inbound mode + empty yard ✓")


def test_mask_outbound_empty_yard():
    """In outbound mode with empty yard, no outbound actions are valid."""
    env = DepotEnv(DepotConfig())
    env.reset(seed=0)
    env.mode = 1  # force outbound

    mask = env.action_masks()
    assert not mask.any(), "Empty yard should have no valid outbound actions"
    print("Mask: outbound mode + empty yard ✓")


def test_mask_inbound_full_stack():
    """A fully occupied stack should be masked out in inbound mode."""
    env = DepotEnv(DepotConfig())
    env.reset(seed=0)
    env.mode = 0

    # Fill stack (0, 0, 0) completely
    env.grid[0, 0, 0, :] = 0  # all tiers occupied with dwell=0

    mask = env.action_masks()
    full_stack_idx = env.encode_inbound(0, 0, 0)
    assert not mask[full_stack_idx], "Full stack should be masked"

    # Adjacent stack should still be valid
    other_idx = env.encode_inbound(0, 0, 1)
    assert mask[other_idx], "Non-full stack should still be valid"
    print("Mask: inbound + full stack ✓")


def test_mask_outbound_occupied_slots():
    """Only occupied slots should be valid in outbound mode."""
    env = DepotEnv(DepotConfig())
    env.reset(seed=0)
    env.mode = 1

    # Place two containers
    env.grid[0, 0, 0, 0] = 5
    env.grid[2, 1, 3, 2] = 1

    mask = env.action_masks()
    idx_a = env.encode_outbound(0, 0, 0, 0)
    idx_b = env.encode_outbound(2, 1, 3, 2)
    idx_empty = env.encode_outbound(0, 0, 0, 1)  # tier 1 empty

    assert mask[idx_a], "Occupied slot (0,0,0,0) should be valid"
    assert mask[idx_b], "Occupied slot (2,1,3,2) should be valid"
    assert not mask[idx_empty], "Empty slot should be masked"
    # No inbound valid in outbound mode
    assert not mask[:env.n_inbound].any(), "Inbound should be fully masked"
    print("Mask: outbound + occupied slots ✓")


if __name__ == "__main__":
    test_encode_decode_inbound_roundtrip()
    test_encode_decode_outbound_roundtrip()
    test_action_indices_partition()
    test_mask_inbound_empty_yard()
    test_mask_outbound_empty_yard()
    test_mask_inbound_full_stack()
    test_mask_outbound_occupied_slots()
    print("\nAll action space tests passed ✓")